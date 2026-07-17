/**
 * perf(search): keyword search runs concurrently with the embed+vector arm.
 *
 * Pre-fix, hybridSearch awaited engine.searchKeyword BEFORE starting any of
 * the vector-arm work (provider probe, expansion, query embed, HNSW search),
 * even though the first joint use of the two arms is RRF fusion. On remote
 * setups that serialized ~0.6s of keyword round-trip in front of the embed.
 * The fix dispatches keyword up front and joins it at its three consumers
 * (no-provider path, vector-failed fallback, fusion).
 *
 * Pins:
 *   - The vector arm starts while keyword is still in flight (bounded
 *     deterministic probe; a sequential regression fails in ~2s, no hang).
 *   - A keyword error still rejects hybridSearch (same contract, new await
 *     site), with no unhandled rejection.
 *   - The embed-failed fallback path still returns keyword rows.
 *
 * PGLite + gateway embed-transport seam; engine methods are wrapped on the
 * INSTANCE (no module mocking). GBRAIN_HOME is pinned to an empty temp dir
 * so a developer's real ~/.gbrain config can't leak into column resolution.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach } from 'bun:test';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { PGLiteEngine } from '../../src/core/pglite-engine.ts';
import { withEnv } from '../helpers/with-env.ts';
import { hybridSearch } from '../../src/core/search/hybrid.ts';
import {
  configureGateway,
  resetGateway,
  __setEmbedTransportForTests,
} from '../../src/core/ai/gateway.ts';
import type { ChunkInput } from '../../src/core/types.ts';

const DIM = 1536;

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

const TEST_HOME = mkdtempSync(join(tmpdir(), 'gbrain-kw-parallel-'));
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

let engine: PGLiteEngine;
const realMethods: { searchKeyword?: PGLiteEngine['searchKeyword']; searchVector?: PGLiteEngine['searchVector'] } = {};

beforeAll(async () => {
  resetGateway();
  configureGateway({
    embedding_model: 'openai:text-embedding-3-large',
    embedding_dimensions: DIM,
    env: { OPENAI_API_KEY: 'sk-fake-kw-parallel' },
  });
  engine = new PGLiteEngine();
  await engine.connect({});
  await engine.initSchema();
  realMethods.searchKeyword = engine.searchKeyword.bind(engine);
  realMethods.searchVector = engine.searchVector.bind(engine);

  // One page with an embedded chunk so both arms return rows.
  await engine.putPage('people/alice-example', {
    type: 'person',
    title: 'Alice Example',
    compiled_truth: 'Alice Example is a test person for keyword-parallel tests.',
  });
  const chunks: ChunkInput[] = [
    {
      chunk_index: 0,
      chunk_text: 'Alice Example is a test person for keyword-parallel tests.',
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

beforeEach(() => {
  __setEmbedTransportForTests((async ({ values }: { values: string[] }) => ({
    embeddings: values.map(() => VEC),
    usage: { tokens: values.length },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  })) as any);
});

afterEach(() => {
  // Restore instance methods any test wrapped.
  engine.searchKeyword = realMethods.searchKeyword!;
  engine.searchVector = realMethods.searchVector!;
});

function search(query: string, opts: Parameters<typeof hybridSearch>[2] = {}) {
  return withEnv({ GBRAIN_HOME: TEST_HOME }, () => hybridSearch(engine, query, opts));
}

describe('hybridSearch: keyword overlaps the vector arm', () => {
  test('vector search starts while keyword is still in flight', async () => {
    let vectorStarted = false;
    let vectorStartedBeforeKeywordResolved = false;

    engine.searchKeyword = async (q, o) => {
      // Resolve only once the vector arm has been DISPATCHED. Sequential
      // code awaits keyword before any vector work → the flag stays false
      // past the 2s bound → the assertion fails (bounded, no hang).
      const start = Date.now();
      while (!vectorStarted && Date.now() - start < 2000) {
        await sleep(5);
      }
      vectorStartedBeforeKeywordResolved = vectorStarted;
      return realMethods.searchKeyword!(q, o);
    };
    engine.searchVector = async (emb, o) => {
      vectorStarted = true;
      return realMethods.searchVector!(emb, o);
    };

    const results = await search('alice example');
    expect(vectorStartedBeforeKeywordResolved).toBe(true);
    expect(results.length).toBeGreaterThan(0);
  });

  test('a keyword error still rejects hybridSearch (no unhandled rejection)', async () => {
    engine.searchKeyword = async () => {
      throw new Error('keyword arm down');
    };
    await expect(search('alice example')).rejects.toThrow('keyword arm down');
  });

  test('embed failure still falls back to keyword rows', async () => {
    __setEmbedTransportForTests((async () => {
      throw new Error('embed provider down');
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    }) as any);

    let meta: import('../../src/core/types.ts').HybridSearchMeta | null = null;
    const results = await search('alice example', { onMeta: (m) => { meta = m; } });
    expect(meta!.vector_enabled).toBe(false);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].slug).toBe('people/alice-example');
  });
});
