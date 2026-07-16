/**
 * perf(search): runPostFusionStages fetches its three metadata inputs
 * concurrently.
 *
 * getBacklinkCounts / getSalienceScores / getEffectiveDates read independent
 * data keyed only on slugs/refs. Pre-fix they were awaited back to back, so
 * every query paid three sequential DB round-trips (~0.5-0.7s on remote
 * setups). The fix dispatches all three fetches up front and then applies
 * the boosts in the original order (backlink → salience → recency), which
 * is what determines score attribution.
 *
 * Pins:
 *   - All three fetches are in flight together (a sequential regression
 *     fails deterministically, no hanging test).
 *   - One rejecting fetch skips ONLY its own boost; the others still apply
 *     (the per-stage non-fatal contract).
 *   - Disabled stages ('off' / applyBacklinks=false) never fetch, and the
 *     empty-results early return fires before any fetch is dispatched.
 *   - All three boosts stamp + apply together when all fetches return data.
 *
 * Pure mock-engine unit test: no PGLite, no provider.
 */

import { describe, test, expect } from 'bun:test';
import { runPostFusionStages } from '../../src/core/search/hybrid.ts';
import type { BrainEngine } from '../../src/core/engine.ts';
import type { SearchResult } from '../../src/core/types.ts';

function makeResult(slug: string, score = 0.5): SearchResult {
  return {
    slug,
    page_id: 1,
    title: `Title for ${slug}`,
    type: 'concept',
    chunk_text: `chunk text for ${slug}`,
    chunk_source: 'compiled_truth',
    chunk_id: 1,
    chunk_index: 0,
    score,
    stale: false,
  };
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

type EngineFns = Pick<BrainEngine, 'getBacklinkCounts' | 'getSalienceScores' | 'getEffectiveDates'>;

function makeEngine(overrides: Partial<EngineFns> = {}): BrainEngine {
  return {
    getBacklinkCounts: async () => new Map<string, number>(),
    getSalienceScores: async () => new Map<string, number>(),
    getEffectiveDates: async () => new Map<string, Date>(),
    ...overrides,
    // Everything else (alias-resolved stage, etc.) is missing on purpose:
    // those stages are fail-soft and swallow the resulting TypeError, same
    // as they swallow a missing table on a pre-v105 brain.
  } as unknown as BrainEngine;
}

// Explicit recency knobs so the boost fires deterministically for any slug
// (the shipped DEFAULT_FALLBACK depends on prefix conventions).
const RECENCY_OPTS = {
  decayMap: {},
  fallback: { halflifeDays: 30, coefficient: 0.3 },
};

describe('runPostFusionStages: concurrent metadata fetches', () => {
  test('all three fetches are in flight before the first resolves', async () => {
    let salienceCalled = false;
    let datesCalled = false;
    let othersInFlightWhenBacklinkResolved = false;

    const engine = makeEngine({
      getBacklinkCounts: async () => {
        // Resolve only once the other two fetches have been DISPATCHED.
        // Sequential code awaits this first → the flags stay false past the
        // 2s bound → the assertion below fails (bounded, no hang).
        const start = Date.now();
        while (!(salienceCalled && datesCalled) && Date.now() - start < 2000) {
          await sleep(5);
        }
        othersInFlightWhenBacklinkResolved = salienceCalled && datesCalled;
        return new Map([['a', 2]]);
      },
      getSalienceScores: async () => {
        salienceCalled = true;
        return new Map([['default::a', 5]]);
      },
      getEffectiveDates: async () => {
        datesCalled = true;
        return new Map([['default::a', new Date(Date.now() - 86_400_000)]]);
      },
    });

    const results = [makeResult('a')];
    await runPostFusionStages(engine, results, {
      applyBacklinks: true,
      salience: 'on',
      recency: 'on',
      ...RECENCY_OPTS,
    });

    expect(othersInFlightWhenBacklinkResolved).toBe(true);
    // The overlapped fetches still feed their boosts.
    expect(results[0].backlink_boost).toBeGreaterThan(1);
    expect(results[0].salience_boost).toBeGreaterThan(1);
    expect(results[0].recency_boost).toBeGreaterThan(1);
  });

  test('a rejecting fetch skips only its own boost (per-stage fail-soft)', async () => {
    const engine = makeEngine({
      getBacklinkCounts: async () => {
        throw new Error('backlink fetch down');
      },
      getSalienceScores: async () => new Map([['default::a', 5]]),
      getEffectiveDates: async () => new Map([['default::a', new Date(Date.now() - 86_400_000)]]),
    });

    const results = [makeResult('a')];
    await runPostFusionStages(engine, results, {
      applyBacklinks: true,
      salience: 'on',
      recency: 'on',
      ...RECENCY_OPTS,
    });

    expect(results[0].backlink_boost).toBeUndefined();
    expect(results[0].salience_boost).toBeGreaterThan(1);
    expect(results[0].recency_boost).toBeGreaterThan(1);
  });

  test('a SYNCHRONOUSLY throwing engine method is as fail-soft as a rejection', async () => {
    // The pre-parallel code wrapped each invocation in try/catch, so an
    // engine whose method throws before returning a promise (a minimal
    // engine or test double with a non-async implementation, or a missing
    // method entirely) never escaped runPostFusionStages. The concurrent
    // kickoff must preserve that: sync throws are converted to a skipped
    // boost, and the OTHER stages still apply.
    const engine = makeEngine({
      getBacklinkCounts: (() => {
        throw new Error('sync throw before any promise exists');
      }) as unknown as EngineFns['getBacklinkCounts'],
      getSalienceScores: async () => new Map([['default::a', 5]]),
      getEffectiveDates: async () => new Map([['default::a', new Date(Date.now() - 86_400_000)]]),
    });

    const results = [makeResult('a')];
    await runPostFusionStages(engine, results, {
      applyBacklinks: true,
      salience: 'on',
      recency: 'on',
      ...RECENCY_OPTS,
    });

    expect(results[0].backlink_boost).toBeUndefined();
    expect(results[0].salience_boost).toBeGreaterThan(1);
    expect(results[0].recency_boost).toBeGreaterThan(1);
  });

  test('an engine missing a metadata method entirely stays fail-soft', async () => {
    // engine.getSalienceScores === undefined makes the call site itself
    // throw a TypeError synchronously. Same contract: skip that boost only.
    const engine = makeEngine({
      getSalienceScores: undefined as unknown as EngineFns['getSalienceScores'],
    });

    const results = [makeResult('a')];
    await runPostFusionStages(engine, results, {
      applyBacklinks: true,
      salience: 'on',
      recency: 'off',
    });

    // Backlink map is empty (no boost recorded) but the stage ran without
    // the missing-salience TypeError escaping.
    expect(results[0].salience_boost).toBeUndefined();
    expect(results[0].score).toBe(results[0].base_score!);
  });

  test('disabled stages never fetch; empty results fetch nothing', async () => {
    let backlinkCalls = 0;
    let salienceCalls = 0;
    let datesCalls = 0;
    const engine = makeEngine({
      getBacklinkCounts: async () => { backlinkCalls++; return new Map(); },
      getSalienceScores: async () => { salienceCalls++; return new Map(); },
      getEffectiveDates: async () => { datesCalls++; return new Map(); },
    });

    await runPostFusionStages(engine, [makeResult('a')], {
      applyBacklinks: false,
      salience: 'off',
      recency: 'off',
    });
    expect(backlinkCalls).toBe(0);
    expect(salienceCalls).toBe(0);
    expect(datesCalls).toBe(0);

    await runPostFusionStages(engine, [], {
      applyBacklinks: true,
      salience: 'on',
      recency: 'on',
    });
    expect(backlinkCalls).toBe(0);
    expect(salienceCalls).toBe(0);
    expect(datesCalls).toBe(0);
  });

  test('boost math is unchanged: applied factors match the sequential formulas', async () => {
    const engine = makeEngine({
      getBacklinkCounts: async () => new Map([['a', 3]]),
      getSalienceScores: async () => new Map([['default::a', 4]]),
      getEffectiveDates: async () => new Map([['default::a', new Date(Date.now() - 30 * 86_400_000)]]),
    });

    const base = 0.5;
    const results = [makeResult('a', base)];
    await runPostFusionStages(engine, results, {
      applyBacklinks: true,
      salience: 'on',
      recency: 'on',
      ...RECENCY_OPTS,
    });

    const backlinkFactor = 1.0 + 0.05 * Math.log(1 + 3);
    const salienceFactor = 1.0 + 0.15 * Math.log(1 + 4);
    // ~30 days old with halflife 30 → coefficient * 30/(30+30) ≈ half strength.
    expect(results[0].backlink_boost).toBeCloseTo(backlinkFactor, 10);
    expect(results[0].salience_boost).toBeCloseTo(salienceFactor, 10);
    expect(results[0].recency_boost).toBeGreaterThan(1);
    expect(results[0].recency_boost!).toBeLessThanOrEqual(1.0 + 0.3);
    expect(results[0].score).toBeCloseTo(
      base * backlinkFactor * salienceFactor * results[0].recency_boost!,
      10,
    );
    expect(results[0].base_score).toBe(base);
  });
});
