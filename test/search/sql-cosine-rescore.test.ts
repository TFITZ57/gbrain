/**
 * perf(search): cosine re-score computed in SQL.
 *
 * cosineReScore previously called engine.getEmbeddingsByChunkIds, pulling
 * N full vectors (e.g. 50 × 1280 float32) across the wire just to reduce
 * each to a single float client-side, 0.6-0.9s of the hybrid re-score
 * stage on remote databases. The new optional engine method
 * getCosineSimilaritiesByChunkIds computes `1 - (column <=> query)` where
 * the data lives and returns only the scores; hybrid.cosineReScore uses it
 * when present and keeps the download path as the fallback.
 *
 * Pins (PGLite engine; the same SQL shape runs on postgres-engine):
 *   - SQL similarities match client-side cosineSimilarity to float precision.
 *   - Chunks with a NULL column are absent from the map (same contract as
 *     getEmbeddingsByChunkIds).
 *   - Zero-magnitude chunk vectors map to 0 (client denom===0 contract),
 *     never NaN.
 *   - The column parameter selects the embedding space (descriptor form,
 *     halfvec-safe), and an unregistered bare string throws BEFORE SQL.
 *   - hybridSearch still works end-to-end through the SQL path, and an
 *     engine WITHOUT the method falls back to the download path.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach } from 'bun:test';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { PGLiteEngine } from '../../src/core/pglite-engine.ts';
import { withEnv } from '../helpers/with-env.ts';
import { hybridSearch, cosineSimilarity } from '../../src/core/search/hybrid.ts';
import { EmbeddingColumnNotRegisteredError } from '../../src/core/search/embedding-column.ts';
import {
  configureGateway,
  resetGateway,
  __setEmbedTransportForTests,
} from '../../src/core/ai/gateway.ts';
import type { BrainEngine } from '../../src/core/engine.ts';
import type { ChunkInput, ResolvedColumn } from '../../src/core/types.ts';

const DIM = 1536;
const TEST_HOME = mkdtempSync(join(tmpdir(), 'gbrain-sql-cosine-'));

function vec(seed: number, dim = DIM): number[] {
  const e = new Array<number>(dim);
  for (let i = 0; i < dim; i++) e[i] = Math.sin(seed + i * 0.01);
  let mag = 0;
  for (let i = 0; i < dim; i++) mag += e[i] * e[i];
  mag = Math.sqrt(mag);
  for (let i = 0; i < dim; i++) e[i] /= mag;
  return e;
}
const QUERY_VEC = vec(1);
const CHUNK_VECS = [vec(2), vec(50), vec(400)];

let engine: PGLiteEngine;
let chunkIds: number[] = [];

async function rawQuery(q: string, params: unknown[] = []): Promise<unknown> {
  return (engine as unknown as { db: { query: (q: string, p: unknown[]) => Promise<unknown> } }).db.query(q, params);
}

beforeAll(async () => {
  resetGateway();
  configureGateway({
    embedding_model: 'openai:text-embedding-3-large',
    embedding_dimensions: DIM,
    env: { OPENAI_API_KEY: 'sk-fake-sql-cosine' },
  });
  engine = new PGLiteEngine();
  await engine.connect({});
  await engine.initSchema();

  // Four chunks: three with distinct embeddings, one left NULL.
  await engine.putPage('concepts/sql-cosine', {
    type: 'concept',
    title: 'SQL Cosine Rescore',
    compiled_truth: 'sql cosine rescore parity fixture',
  });
  const chunks: ChunkInput[] = [0, 1, 2, 3].map((i) => ({
    chunk_index: i,
    chunk_text: `sql cosine rescore parity fixture chunk ${i}`,
    chunk_source: 'compiled_truth',
  }));
  await engine.upsertChunks('concepts/sql-cosine', chunks);
  const rows = (await rawQuery(
    `SELECT cc.id FROM content_chunks cc JOIN pages p ON p.id = cc.page_id
      WHERE p.slug = 'concepts/sql-cosine' ORDER BY cc.chunk_index`,
  )) as { rows: Array<{ id: number }> };
  chunkIds = rows.rows.map((r) => r.id);
  for (let i = 0; i < 3; i++) {
    await rawQuery(`UPDATE content_chunks SET embedding = $1::vector WHERE id = $2`, [
      `[${CHUNK_VECS[i].join(',')}]`,
      chunkIds[i],
    ]);
  }
  // chunkIds[3] stays NULL.

  // Ad-hoc alternate column at a different dim, mimicking a per-instance
  // registered column (same fixture shape as cosine-rescore-column.test.ts).
  await rawQuery(`ALTER TABLE content_chunks ADD COLUMN IF NOT EXISTS embedding_alt vector(8)`);
  await rawQuery(`UPDATE content_chunks SET embedding_alt = $1::vector WHERE id = $2`, [
    `[1,0,0,0,0,0,0,0]`,
    chunkIds[0],
  ]);
});

afterAll(async () => {
  await engine.disconnect();
  __setEmbedTransportForTests(null);
  resetGateway();
});

beforeEach(() => {
  __setEmbedTransportForTests((async ({ values }: { values: string[] }) => ({
    embeddings: values.map(() => QUERY_VEC),
    usage: { tokens: values.length },
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  })) as any);
});

describe('getCosineSimilaritiesByChunkIds: parity with client-side cosine', () => {
  test('SQL similarities match cosineSimilarity to float precision', async () => {
    const q = new Float32Array(QUERY_VEC);
    const map = await engine.getCosineSimilaritiesByChunkIds(chunkIds, q);
    expect(map.size).toBe(3); // NULL-embedding chunk absent

    const stored = await engine.getEmbeddingsByChunkIds(chunkIds);
    for (let i = 0; i < 3; i++) {
      const expected = cosineSimilarity(q, stored.get(chunkIds[i])!);
      expect(map.get(chunkIds[i])!).toBeCloseTo(expected, 5);
    }
    expect(map.has(chunkIds[3])).toBe(false);
  });

  test('empty id list short-circuits to an empty map', async () => {
    const map = await engine.getCosineSimilaritiesByChunkIds([], new Float32Array(QUERY_VEC));
    expect(map.size).toBe(0);
  });

  test('zero-magnitude chunk vector maps to 0, never NaN', async () => {
    const zeroed = chunkIds[2];
    const zeroVec = new Array(DIM).fill(0);
    await rawQuery(`UPDATE content_chunks SET embedding = $1::vector WHERE id = $2`, [
      `[${zeroVec.join(',')}]`,
      zeroed,
    ]);
    try {
      const map = await engine.getCosineSimilaritiesByChunkIds([zeroed], new Float32Array(QUERY_VEC));
      expect(map.get(zeroed)).toBe(0);
    } finally {
      await rawQuery(`UPDATE content_chunks SET embedding = $1::vector WHERE id = $2`, [
        `[${CHUNK_VECS[2].join(',')}]`,
        zeroed,
      ]);
    }
  });

  test('descriptor column parameter selects the alternate embedding space', async () => {
    const altCol: ResolvedColumn = {
      name: 'embedding_alt',
      type: 'vector',
      dimensions: 8,
      embeddingModel: 'openai:test-alt',
    };
    const q8 = new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]);
    const map = await engine.getCosineSimilaritiesByChunkIds(chunkIds, q8, altCol);
    // Only chunk 0 has embedding_alt populated; identical vector → cosine 1.
    expect(map.size).toBe(1);
    expect(map.get(chunkIds[0])!).toBeCloseTo(1.0, 6);
  });

  test('unregistered bare string column throws before any SQL', async () => {
    await expect(
      engine.getCosineSimilaritiesByChunkIds(chunkIds, new Float32Array(QUERY_VEC), 'embedding_alt'),
    ).rejects.toThrow(EmbeddingColumnNotRegisteredError);
  });
});

describe('cosineReScore path selection through hybridSearch', () => {
  function search(eng: BrainEngine, query: string) {
    return withEnv({ GBRAIN_HOME: TEST_HOME }, () => hybridSearch(eng, query, {}));
  }

  test('end-to-end hybridSearch works through the SQL re-score path', async () => {
    let sqlCalls = 0;
    const orig = engine.getCosineSimilaritiesByChunkIds!.bind(engine);
    engine.getCosineSimilaritiesByChunkIds = async (ids, emb, col) => {
      sqlCalls++;
      return orig(ids, emb, col);
    };
    try {
      const results = await search(engine, 'sql cosine rescore parity fixture');
      expect(results.length).toBeGreaterThan(0);
      expect(sqlCalls).toBe(1);
    } finally {
      engine.getCosineSimilaritiesByChunkIds = orig;
    }
  });

  test('engine without the method falls back to the vector-download path', async () => {
    let downloadCalls = 0;
    const facade = new Proxy(engine, {
      get(target, prop, receiver) {
        if (prop === 'getCosineSimilaritiesByChunkIds') return undefined;
        if (prop === 'getEmbeddingsByChunkIds') {
          return async (ids: number[], column?: string) => {
            downloadCalls++;
            return target.getEmbeddingsByChunkIds(ids, column);
          };
        }
        const value = Reflect.get(target, prop, target);
        return typeof value === 'function' ? value.bind(target) : value;
      },
    }) as unknown as BrainEngine;

    const results = await search(facade, 'sql cosine rescore parity fixture');
    expect(results.length).toBeGreaterThan(0);
    expect(downloadCalls).toBe(1);
  });
});

describe('halfvec columns: float-precision score parity', () => {
  // searchVector's halfvec cast fragment quantizes the QUERY to half
  // precision before the distance. The re-score contract is parity with the
  // client-side path, which cosines the Float32 query against the stored
  // (already-quantized) values; the SQL path must therefore upcast the
  // stored halfvec to vector and bind the query as a float vector. These
  // pins prove both that the shipped SQL matches the client math to float
  // precision AND that the quantized-query alternative measurably differs
  // (so the parity assertion is actually sensitive to the cast choice).
  const HDIM = 8;
  const HCOL = { name: 'embedding_half', type: 'halfvec' as const, dimensions: HDIM, embeddingModel: '' };
  let halfIds: number[] = [];
  let queryHalf: Float32Array;

  function parseVectorText(text: string): Float32Array {
    return new Float32Array(text.replace(/^\[|\]$/g, '').split(',').map(Number));
  }

  beforeAll(async () => {
    queryHalf = new Float32Array(vec(7, HDIM));
    await rawQuery(`ALTER TABLE content_chunks ADD COLUMN IF NOT EXISTS embedding_half halfvec(${HDIM})`);
    halfIds = chunkIds.slice(0, 3);
    for (let i = 0; i < 3; i++) {
      await rawQuery(`UPDATE content_chunks SET embedding_half = $1::halfvec(${HDIM}) WHERE id = $2`, [
        `[${vec(20 + i, HDIM).join(',')}]`,
        halfIds[i],
      ]);
    }
  });

  test('SQL halfvec scores match Float32-query client cosine to float precision', async () => {
    const map = await engine.getCosineSimilaritiesByChunkIds(halfIds, queryHalf, HCOL);
    expect(map.size).toBe(3);

    for (const id of halfIds) {
      // The client-parity reference: Float32 query against the stored,
      // already-quantized values (read back dequantized from the column).
      const row = (await rawQuery(
        `SELECT embedding_half::vector::text AS v FROM content_chunks WHERE id = $1`,
        [id],
      )) as { rows: Array<{ v: string }> };
      const stored = parseVectorText(row.rows[0].v);
      const expected = cosineSimilarity(queryHalf, stored);
      expect(Math.abs(map.get(id)! - expected)).toBeLessThan(1e-6);
    }
  });

  test('quantizing the query (the rejected cast) measurably diverges', async () => {
    // Sensitivity guard for the parity pin above: if half-quantizing the
    // query made no observable difference on these vectors, the previous
    // test could not distinguish the two casts. Compute the quantized-query
    // cosine the old fragment would have produced and require a gap well
    // above the parity tolerance for at least one chunk.
    const qq = (await rawQuery(
      `SELECT ($1::halfvec(${HDIM}))::vector::text AS v`,
      [`[${Array.from(queryHalf).join(',')}]`],
    )) as { rows: Array<{ v: string }> };
    const quantizedQuery = parseVectorText(qq.rows[0].v);

    let maxGap = 0;
    for (const id of halfIds) {
      const row = (await rawQuery(
        `SELECT embedding_half::vector::text AS v FROM content_chunks WHERE id = $1`,
        [id],
      )) as { rows: Array<{ v: string }> };
      const stored = parseVectorText(row.rows[0].v);
      const gap = Math.abs(
        cosineSimilarity(quantizedQuery, stored) - cosineSimilarity(queryHalf, stored),
      );
      maxGap = Math.max(maxGap, gap);
    }
    expect(maxGap).toBeGreaterThan(1e-5);
  });
});
