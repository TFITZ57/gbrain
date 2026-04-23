import { afterEach, beforeEach, describe, expect, test } from 'bun:test';

import { embedBatch } from '../src/core/embedding.ts';

const originalFetch = globalThis.fetch;
const originalProvider = process.env.GBRAIN_EMBED_PROVIDER;
const originalModel = process.env.GBRAIN_EMBED_MODEL;
const originalDimensions = process.env.GBRAIN_EMBED_DIMENSIONS;
const originalMaxChars = process.env.GBRAIN_OLLAMA_MAX_CHARS;

beforeEach(() => {
  process.env.GBRAIN_EMBED_PROVIDER = 'ollama';
  process.env.GBRAIN_EMBED_MODEL = 'mxbai-embed-large';
  process.env.GBRAIN_EMBED_DIMENSIONS = '4';
  delete process.env.GBRAIN_OLLAMA_MAX_CHARS;
});

afterEach(() => {
  globalThis.fetch = originalFetch;

  if (originalProvider === undefined) delete process.env.GBRAIN_EMBED_PROVIDER;
  else process.env.GBRAIN_EMBED_PROVIDER = originalProvider;

  if (originalModel === undefined) delete process.env.GBRAIN_EMBED_MODEL;
  else process.env.GBRAIN_EMBED_MODEL = originalModel;

  if (originalDimensions === undefined) delete process.env.GBRAIN_EMBED_DIMENSIONS;
  else process.env.GBRAIN_EMBED_DIMENSIONS = originalDimensions;

  if (originalMaxChars === undefined) delete process.env.GBRAIN_OLLAMA_MAX_CHARS;
  else process.env.GBRAIN_OLLAMA_MAX_CHARS = originalMaxChars;
});

describe('embedding ollama fallback', () => {
  test('shrinks context-length failures until the request fits', async () => {
    const seenLengths: number[] = [];

    globalThis.fetch = (async (_url: string | URL | Request, init?: RequestInit) => {
      const body = JSON.parse(String(init?.body ?? '{}')) as { input?: string[] };
      const input = body.input?.[0] ?? '';
      seenLengths.push(input.length);

      if (input.length > 1000) {
        return new Response(
          JSON.stringify({ error: 'the input length exceeds the context length' }),
          { status: 400, headers: { 'Content-Type': 'application/json' } },
        );
      }

      return new Response(
        JSON.stringify({ embeddings: [[0.1, 0.2, 0.3, 0.4]] }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      );
    }) as typeof fetch;

    const result = await embedBatch(['x'.repeat(1600)]);

    expect(result).toHaveLength(1);
    const values = Array.from(result[0]).map(value => Number(value.toFixed(3)));
    expect(values).toEqual([0.1, 0.2, 0.3, 0.4]);
    expect(seenLengths[0]).toBe(1500);
    expect(seenLengths.length).toBeGreaterThan(1);
    expect(seenLengths[seenLengths.length - 1]).toBeLessThanOrEqual(1000);
  });
});
