import { afterEach, describe, expect, test } from 'bun:test';
import { mkdtempSync, readFileSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

import { buildCliCommandTelemetry, parseHermesCallerContext } from '../src/core/cli-telemetry.ts';

describe('cli telemetry helpers', () => {
  test('parseHermesCallerContext returns null for invalid json', () => {
    expect(parseHermesCallerContext('{not-json')).toBeNull();
  });

  test('buildCliCommandTelemetry preserves canonical command data', () => {
    const telemetry = buildCliCommandTelemetry({
      rawCommand: 'ask',
      canonicalCommand: 'query',
      commandKind: 'operation',
      subArgs: ['who is Tyler', '--no-expand'],
      aliasApplied: 'ask',
      callerContext: {
        caller: 'hermes-terminal-tool',
        taskId: 'task-1',
        platform: 'discord',
      },
    });

    expect(telemetry.name).toBe('gbrain:query');
    expect(telemetry.commandRaw).toBe('ask');
    expect(telemetry.commandCanonical).toBe('query');
    expect(telemetry.commandKind).toBe('operation');
    expect(telemetry.aliasApplied).toBe('ask');
    expect(telemetry.callerContext?.caller).toBe('hermes-terminal-tool');
    expect(telemetry.subArgs).toEqual(['who is Tyler', '--no-expand']);
  });
});

describe('cli telemetry integration', () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length) {
      const dir = tempDirs.pop();
      if (dir) rmSync(dir, { recursive: true, force: true });
    }
  });

  test('--version writes canonical command span log when requested', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'gbrain-cli-telemetry-'));
    tempDirs.push(dir);
    const logPath = join(dir, 'telemetry.jsonl');

    const proc = Bun.spawn(['bun', 'run', 'src/cli.ts', '--version'], {
      cwd: new URL('..', import.meta.url).pathname,
      stdout: 'pipe',
      stderr: 'pipe',
      env: {
        ...process.env,
        GBRAIN_CLI_TELEMETRY_LOG_PATH: logPath,
        HERMES_GBRAIN_SPAN_CONTEXT: JSON.stringify({
          caller: 'hermes-terminal-tool',
          taskId: 'task-42',
          platform: 'discord',
        }),
      },
    });

    const stdout = await new Response(proc.stdout).text();
    const stderr = await new Response(proc.stderr).text();
    const exitCode = await proc.exited;

    expect(stdout.trim()).toMatch(/^gbrain \d+\.\d+\.\d+/);
    expect(stderr).toBe('');
    expect(exitCode).toBe(0);

    const entries = readFileSync(logPath, 'utf-8')
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line));

    expect(entries).toHaveLength(1);
    expect(entries[0].name).toBe('gbrain:version');
    expect(entries[0].commandCanonical).toBe('version');
    expect(entries[0].commandKind).toBe('builtin');
    expect(entries[0].callerContext).toMatchObject({
      caller: 'hermes-terminal-tool',
      taskId: 'task-42',
      platform: 'discord',
    });
    expect(entries[0].exitCode).toBe(0);
    expect(typeof entries[0].durationMs).toBe('number');
  });
});
