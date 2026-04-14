/**
 * Embedding Service
 *
 * Default provider is OpenAI.
 * Set GBRAIN_EMBED_PROVIDER=ollama to use a local Ollama embedding model.
 * Ollama embeddings are normalized to the configured vector size so the
 * disposable pilot can keep the existing 1536-dim schema.
 */

import OpenAI from 'openai';

const OPENAI_MAX_CHARS = 8000;
const OLLAMA_DEFAULT_MAX_CHARS = 1500;
const OLLAMA_MIN_CHARS = 400;
const OLLAMA_SHRINK_RATIO = 0.75;
const MAX_RETRIES = 5;
const BASE_DELAY_MS = 4000;
const MAX_DELAY_MS = 120000;
const BATCH_SIZE = 100;
const OLLAMA_EMBED_URL = process.env.OLLAMA_EMBED_URL || 'http://127.0.0.1:11434/api/embed';

let client: OpenAI | null = null;

function getProvider(): string {
  return (process.env.GBRAIN_EMBED_PROVIDER || 'openai').toLowerCase();
}

function getModel(provider = getProvider()): string {
  return process.env.GBRAIN_EMBED_MODEL || (provider === 'ollama' ? 'nomic-embed-text' : 'text-embedding-3-large');
}

function getDimensions(): number {
  return parseInt(process.env.GBRAIN_EMBED_DIMENSIONS || '1536', 10);
}

function getMaxChars(provider = getProvider()): number {
  const raw = process.env.GBRAIN_EMBED_MAX_CHARS
    || (provider === 'ollama' ? process.env.GBRAIN_OLLAMA_MAX_CHARS : undefined);
  if (raw) {
    const parsed = parseInt(raw, 10);
    if (!Number.isNaN(parsed) && parsed > 0) return parsed;
  }
  return provider === 'ollama' ? OLLAMA_DEFAULT_MAX_CHARS : OPENAI_MAX_CHARS;
}

function getClient(): OpenAI {
  if (!client) {
    client = new OpenAI();
  }
  return client;
}

function normalizeEmbedding(values: ArrayLike<number>): Float32Array {
  const dimensions = getDimensions();
  const out = new Float32Array(dimensions);
  const limit = Math.min(values.length, dimensions);
  for (let i = 0; i < limit; i++) out[i] = values[i];
  return out;
}

export async function embed(text: string): Promise<Float32Array> {
  const result = await embedBatch([text]);
  return result[0];
}

export async function embedBatch(texts: string[]): Promise<Float32Array[]> {
  const provider = getProvider();
  const truncated = texts.map(t => t.slice(0, getMaxChars(provider)));

  if (provider === 'ollama') {
    return await embedBatchOllama(truncated);
  }

  const results: Float32Array[] = [];
  for (let i = 0; i < truncated.length; i += BATCH_SIZE) {
    const batch = truncated.slice(i, i + BATCH_SIZE);
    const batchResults = await embedBatchWithRetry(batch);
    results.push(...batchResults);
  }

  return results;
}

async function embedBatchWithRetry(texts: string[]): Promise<Float32Array[]> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      return await embedBatchOpenAI(texts);
    } catch (e: unknown) {
      if (attempt === MAX_RETRIES - 1) throw e;

      let delay = exponentialDelay(attempt);

      if (e instanceof OpenAI.APIError && e.status === 429) {
        const retryAfter = e.headers?.['retry-after'];
        if (retryAfter) {
          const parsed = parseInt(retryAfter, 10);
          if (!isNaN(parsed)) {
            delay = parsed * 1000;
          }
        }
      }

      await sleep(delay);
    }
  }

  throw new Error('Embedding failed after all retries');
}

async function embedBatchOpenAI(texts: string[]): Promise<Float32Array[]> {
  const response = await getClient().embeddings.create({
    model: getModel('openai'),
    input: texts,
    dimensions: getDimensions(),
  });

  const sorted = response.data.sort((a, b) => a.index - b.index);
  return sorted.map(d => normalizeEmbedding(d.embedding));
}

async function embedBatchOllama(texts: string[]): Promise<Float32Array[]> {
  const results: Float32Array[] = [];
  for (const text of texts) {
    results.push(await embedSingleOllama(text));
  }
  return results;
}

async function embedSingleOllama(text: string): Promise<Float32Array> {
  let candidate = text.slice(0, getMaxChars('ollama'));

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const embeddings = await requestOllamaEmbeddings([candidate]);
      return embeddings[0];
    } catch (e: unknown) {
      if (isOllamaContextLengthError(e) && candidate.length > OLLAMA_MIN_CHARS) {
        const nextLength = Math.max(OLLAMA_MIN_CHARS, Math.floor(candidate.length * OLLAMA_SHRINK_RATIO));
        if (nextLength >= candidate.length) throw e;
        candidate = candidate.slice(0, nextLength);
        continue;
      }

      if (attempt === MAX_RETRIES - 1) throw e;
      await sleep(exponentialDelay(attempt));
    }
  }

  throw new Error('Ollama embedding failed after all retries');
}

async function requestOllamaEmbeddings(texts: string[]): Promise<Float32Array[]> {
  const response = await fetch(OLLAMA_EMBED_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: getModel('ollama'),
      input: texts,
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Ollama embed failed (${response.status}): ${body}`);
  }

  const json = await response.json() as { embeddings?: number[][] };
  const embeddings = json.embeddings || [];
  if (embeddings.length !== texts.length) {
    throw new Error(`Ollama embed count mismatch: expected ${texts.length}, got ${embeddings.length}`);
  }
  return embeddings.map(values => normalizeEmbedding(values));
}

function isOllamaContextLengthError(err: unknown): boolean {
  return err instanceof Error && /input length exceeds the context length/i.test(err.message);
}

function exponentialDelay(attempt: number): number {
  const delay = BASE_DELAY_MS * Math.pow(2, attempt);
  return Math.min(delay, MAX_DELAY_MS);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export const EMBEDDING_MODEL = getModel();
export const EMBEDDING_DIMENSIONS = getDimensions();
