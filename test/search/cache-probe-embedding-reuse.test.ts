/**
 * perf(search): cache-probe embedding reuse on the cache-miss path.
 *
 * `hybridSearchCached` embeds the query once to probe the semantic cache.
 * Pre-fix, a cache MISS then called the inner `hybridSearch` WITHOUT the
 * embedding, so the identical query string was embedded a second time,
 * a pure duplicate provider round-trip (0.3–0.8s per miss on remote
 * setups). The fix threads the probe embedding through the internal
 * `_cacheProbeEmbedding` opt.
 *
 * Pins:
 *   - Cache miss embeds the query exactly ONCE (probe only; inner reuses).
 *   - Expansion variants still embed fresh (only the original string is
 *     reused).
 *   - Cache hit path still embeds once (the probe) and returns cached rows.
 *   - Direct `hybridSearch` callers (no wrapper) are unchanged: one embed.
 *
 * Uses PGLite in-memory + the gateway embed-transport test seam, so no
 * real provider is hit and every embed call is observable. GBRAIN_HOME is
 * pinned to an empty temp dir (via withEnv) so a developer's real
 * ~/.gbrain/config.json can't leak a different embedding space into the
 * column resolver, so the test behaves identically locally and in CI.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach } from 'bun:test';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { PGLiteEngine } from '../../src/core/pglite-engine.ts';
import { withEnv } from '../helpers/with-env.ts';
import {
  hybridSearch,
  hybridSearchCached,
  awaitPendingSearchCacheWrites,
  _resetPendingSearchCacheWritesForTests,
} from '../../src/core/search/hybrid.ts';
import {
  configureGateway,
  resetGateway,
  __setEmbedTransportForTests,
} from '../../src/core/ai/gateway.ts';
import type { ChunkInput, HybridSearchMeta } from '../../src/core/types.ts';

const DIM = 1536;

// Deterministic, normalized, non-degenerate vector: identical strings must
// produce identical embeddings so the second call's probe matches the row
// the first call stored (cosine 1.0 > 0.92 threshold).
function fakeVec(): number[] {
  const e = new Array<number>(DIM);
  for (let i = 0; i < DIM; i++) e[i] = Math.sin(1 + i * 0.01);
  let mag = 0;
  for (let i = 0; i < DIM; i++) mag += e[i] * e[i];
  mag = Math.sqrt(mag);
  for (let i = 0; i < DIM; i++) e[i] /= mag;
  return e;
}
const VEC = fakeVec();

// Empty config home: hybridSearchCached resolves its embedding column from
// file+DB config; pointing GBRAIN_HOME at an empty dir means "no file
// config", so the gateway pin below is the single source of truth.
const TEST_HOME = mkdtempSync(join(tmpdir(), 'gbrain-probe-reuse-'));

let engine: PGLiteEngine;
let embeddedValues: string[] = [];

beforeAll(async () => {
  // Pin the gateway BEFORE initSchema so the embedding + query_cache columns
  // are sized at DIM regardless of cross-file gateway state (same hermeticity
  // rationale as query-cache.test.ts).
  resetGateway();
  configureGateway({
    embedding_model: 'openai:text-embedding-3-large',
    embedding_dimensions: DIM,
    env: { OPENAI_API_KEY: 'sk-fake-probe-reuse' },
  });
  engine = new PGLiteEngine();
  await engine.connect({});
  await engine.initSchema();

  // Seed one page with an embedded chunk so BOTH recall arms return rows:
  // keyword via chunk text, vector via the planted embedding. Seeded once:
  // the per-test reset below only wipes the query cache (the fixture is
  // read-only for every test in this file; same pattern as query-cache.test.ts).
  await engine.putPage('people/alice-example', {
    type: 'person',
    title: 'Alice Example',
    compiled_truth: 'Alice Example is a test person for probe-reuse tests.',
  });
  const chunks: ChunkInput[] = [
    {
      chunk_index: 0,
      chunk_text: 'Alice Example is a test person for probe-reuse tests.',
      chunk_source: 'compiled_truth',
    },
  ];
  await engine.upsertChunks('people/alice-example', chunks);
  await (engine as unknown as { db: { query: (q: string, p: unknown[]) => Promise<unknown> } }).db.query(
    `UPDATE content_chunks SET embedding = $1::vector`,
    [`[${VEC.join(',')}]`],
  );
});

afterAll(async () => {
  await engine.disconnect();
  __setEmbedTransportForTests(null);
  resetGateway();
});

beforeEach(async () => {
  // Wipe the query cache between tests so ordering doesn't matter; the page
  // fixture itself is read-only for every test in this file.
  await engine.executeRaw(`DELETE FROM query_cache`);
  _resetPendingSearchCacheWritesForTests();
  embeddedValues = [];
  __setEmbedTransportForTests((async ({ values }: { values: string[] }) => {
    embeddedValues.push(...values);
    return {
      embeddings: values.map(() => VEC),
      usage: { tokens: values.length },
    };
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  }) as any);
});

function searchCached(
  query: string,
  opts: Parameters<typeof hybridSearchCached>[2] = {},
): Promise<{ results: Awaited<ReturnType<typeof hybridSearchCached>>; meta: HybridSearchMeta | null }> {
  return withEnv({ GBRAIN_HOME: TEST_HOME }, async () => {
    let meta: HybridSearchMeta | null = null;
    const results = await hybridSearchCached(engine, query, {
      ...opts,
      onMeta: (m) => { meta = m; },
    });
    return { results, meta };
  });
}

describe('hybridSearchCached: cache miss embeds the query exactly once', () => {
  test('miss path reuses the probe embedding (no duplicate embed)', async () => {
    const { results, meta } = await searchCached('alice example');
    expect(meta!.cache?.status).toBe('miss');
    expect(meta!.vector_enabled).toBe(true);
    expect(results.length).toBeGreaterThan(0);
    // Pre-fix this was ['alice example', 'alice example'] (probe + inner
    // re-embed of the same string).
    expect(embeddedValues).toEqual(['alice example']);
  });

  test('expansion variants still embed fresh; original is reused', async () => {
    const { meta } = await searchCached('alice example', {
      expansion: true,
      expandFn: async (q) => [q, `${q} person`, `who is ${q}`],
    });
    expect(meta!.cache?.status).toBe('miss');
    expect(meta!.expansion_applied).toBe(true);
    // Probe (original) + the 2 variants. Pre-fix: 4 embeds (original twice).
    expect(embeddedValues.length).toBe(3);
    expect([...embeddedValues].sort()).toEqual([
      'alice example',
      'alice example person',
      'who is alice example',
    ].sort());
  });

  test('second identical query is a hit: one probe embed, cached results', async () => {
    const { results: first } = await searchCached('alice example');
    expect(first.length).toBeGreaterThan(0);
    await awaitPendingSearchCacheWrites();
    embeddedValues = [];

    const { results: second, meta } = await searchCached('alice example');
    expect(meta!.cache?.status).toBe('hit');
    expect(second.length).toBe(first.length);
    // The hit path needs exactly the probe embed, nothing else.
    expect(embeddedValues).toEqual(['alice example']);
  });
});

describe('bare hybridSearch: no wrapper, unchanged single embed', () => {
  test('direct callers embed once (no probe, no reuse surface)', async () => {
    const results = await withEnv({ GBRAIN_HOME: TEST_HOME }, () =>
      hybridSearch(engine, 'alice example', {}),
    );
    expect(results.length).toBeGreaterThan(0);
    expect(embeddedValues).toEqual(['alice example']);
  });
});

describe('probe reuse is gated on embedding-space match (config drift guard)', () => {
  // The column config lives in the DB, so hybridSearchCached's config read
  // and the inner hybridSearch's read can disagree if config changes between
  // them. The probe carries the embed opts it was computed with; the inner
  // path must discard it and embed fresh when its own resolution differs,
  // otherwise a vector from space A gets searched against space B. These
  // call hybridSearch directly with a hand-built probe to pin the guard on
  // both sides without needing to race a live config write.

  test('matching probe opts: vector is reused, no fresh embed of the original', async () => {
    const results = await withEnv({ GBRAIN_HOME: TEST_HOME }, () =>
      hybridSearch(engine, 'alice example', {
        _cacheProbeEmbedding: {
          embedding: new Float32Array(VEC),
          // The same opts this environment's inner resolution derives (the
          // gateway-pinned model + DIM), so the guard sees a space match.
          embedOpts: { embeddingModel: 'openai:text-embedding-3-large', dimensions: DIM },
        },
      }),
    );
    expect(results.length).toBeGreaterThan(0);
    expect(embeddedValues).toEqual([]);
  });

  test('mismatched probe opts: probe is discarded, original embeds fresh', async () => {
    const staleProbe = new Float32Array(DIM); // wrong-space vector; must not be searched
    const results = await withEnv({ GBRAIN_HOME: TEST_HOME }, () =>
      hybridSearch(engine, 'alice example', {
        _cacheProbeEmbedding: {
          embedding: staleProbe,
          // Claims a different model than this call resolves (undefined
          // opts here): the drift case the guard exists for.
          embedOpts: { embeddingModel: 'openai:text-embedding-3-small', dimensions: DIM },
        },
      }),
    );
    expect(results.length).toBeGreaterThan(0);
    expect(embeddedValues).toEqual(['alice example']);
  });

  test('mismatched dimensions alone also discard the probe', async () => {
    await withEnv({ GBRAIN_HOME: TEST_HOME }, () =>
      hybridSearch(engine, 'alice example', {
        _cacheProbeEmbedding: {
          embedding: new Float32Array(VEC),
          // Same model as the inner resolution; only dimensions differ.
          embedOpts: { embeddingModel: 'openai:text-embedding-3-large', dimensions: DIM / 2 },
        },
      }),
    );
    expect(embeddedValues).toEqual(['alice example']);
  });
});
