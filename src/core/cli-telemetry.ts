import { appendFileSync, mkdirSync } from 'fs';
import { dirname } from 'path';

export type CliCommandKind = 'builtin' | 'cli_only' | 'operation' | 'unknown';

export interface HermesCallerContext {
  caller?: string;
  taskId?: string;
  platform?: string;
  chatId?: string;
  threadId?: string;
  userId?: string;
  sessionKey?: string;
  source?: string;
}

export interface CliCommandTelemetryInput {
  rawCommand: string;
  canonicalCommand: string;
  commandKind: CliCommandKind;
  subArgs: string[];
  aliasApplied?: string;
  callerContext?: HermesCallerContext | null;
}

export interface CliCommandTelemetry {
  name: string;
  commandRaw: string;
  commandCanonical: string;
  commandKind: CliCommandKind;
  subArgs: string[];
  aliasApplied?: string;
  callerContext?: HermesCallerContext | null;
  cwd: string;
  startedAt: string;
}

export interface CliCommandTelemetryRecord extends CliCommandTelemetry {
  exitCode: number;
  durationMs: number;
  finishedAt: string;
  error?: string | null;
}

export function parseHermesCallerContext(raw = process.env.HERMES_GBRAIN_SPAN_CONTEXT): HermesCallerContext | null {
  if (!raw || !raw.trim()) return null;
  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
    const clean: HermesCallerContext = {};
    for (const key of ['caller', 'taskId', 'platform', 'chatId', 'threadId', 'userId', 'sessionKey', 'source'] as const) {
      const value = parsed[key];
      if (typeof value === 'string' && value.trim()) clean[key] = value;
    }
    return Object.keys(clean).length > 0 ? clean : null;
  } catch {
    return null;
  }
}

export function resolveCliCommandIdentity(
  rawCommand: string | undefined,
  canonicalCommand: string | undefined,
  cliOnlyCommands: Set<string>,
  operationCommands: Set<string>,
): { commandCanonical: string; commandKind: CliCommandKind } {
  if (!rawCommand || rawCommand === '--help' || rawCommand === '-h' || rawCommand === 'help') {
    return { commandCanonical: 'help', commandKind: 'builtin' };
  }
  if (canonicalCommand === '--version' || canonicalCommand === 'version') {
    return { commandCanonical: 'version', commandKind: 'builtin' };
  }
  if (canonicalCommand === '--tools-json') {
    return { commandCanonical: 'tools-json', commandKind: 'builtin' };
  }
  const normalized = canonicalCommand || rawCommand;
  if (cliOnlyCommands.has(normalized)) {
    return { commandCanonical: normalized, commandKind: 'cli_only' };
  }
  if (operationCommands.has(normalized)) {
    return { commandCanonical: normalized, commandKind: 'operation' };
  }
  return { commandCanonical: normalized, commandKind: 'unknown' };
}

export function buildCliCommandTelemetry(input: CliCommandTelemetryInput): CliCommandTelemetry {
  return {
    name: `gbrain:${input.canonicalCommand}`,
    commandRaw: input.rawCommand,
    commandCanonical: input.canonicalCommand,
    commandKind: input.commandKind,
    subArgs: input.subArgs,
    aliasApplied: input.aliasApplied,
    callerContext: input.callerContext,
    cwd: process.cwd(),
    startedAt: new Date().toISOString(),
  };
}

function telemetryLogPath(): string {
  return (process.env.GBRAIN_CLI_TELEMETRY_LOG_PATH || '').trim();
}

function sanitizeTag(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9:_-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 80) || 'unknown';
}

function buildTelemetryRecord(
  base: CliCommandTelemetry,
  options: { exitCode: number; startedAtMs: number; error: unknown },
): CliCommandTelemetryRecord {
  const { exitCode, startedAtMs, error } = options;
  return {
    ...base,
    exitCode,
    durationMs: Math.max(0, Date.now() - startedAtMs),
    finishedAt: new Date().toISOString(),
    error: error instanceof Error ? error.message : (typeof error === 'string' ? error : null),
  };
}

function appendTelemetryLog(record: CliCommandTelemetryRecord): void {
  const path = telemetryLogPath();
  if (!path) return;
  mkdirSync(dirname(path), { recursive: true });
  appendFileSync(path, JSON.stringify(record) + '\n', 'utf-8');
}

async function emitOpikCommandSpan(record: CliCommandTelemetryRecord): Promise<void> {
  if (process.env.GBRAIN_CLI_DISABLE_OPIK === '1') return;
  if (!process.env.OPIK_API_KEY) return;

  try {
    const opikModule = await import('opik');
    const { Opik } = opikModule;
    if (typeof opikModule.disableLogger === 'function') {
      opikModule.disableLogger();
    } else if (typeof opikModule.setLoggerLevel === 'function') {
      opikModule.setLoggerLevel('ERROR');
    }
    const projectName = (process.env.GBRAIN_CLI_OPIK_PROJECT_NAME || process.env.OPIK_PROJECT_NAME || 'gbrain-cli').trim();
    const client = new Opik({
      apiKey: process.env.OPIK_API_KEY,
      apiUrl: process.env.OPIK_URL_OVERRIDE || undefined,
      workspaceName: process.env.OPIK_WORKSPACE || undefined,
      projectName,
    });

    const trace = client.trace({
      name: 'gbrain-cli',
      projectName,
      input: {
        command: record.commandCanonical,
        rawCommand: record.commandRaw,
        subArgs: record.subArgs,
      },
      metadata: {
        commandKind: record.commandKind,
        aliasApplied: record.aliasApplied || null,
        callerContext: record.callerContext || null,
        exitCode: record.exitCode,
        durationMs: record.durationMs,
        cwd: record.cwd,
      } as any,
      tags: [
        'gbrain-cli',
        `command:${sanitizeTag(record.commandCanonical)}`,
        `kind:${record.commandKind}`,
        ...(record.callerContext?.caller ? [`caller:${sanitizeTag(record.callerContext.caller)}`] : []),
      ],
    });

    const span = trace.span({
      name: record.name,
      type: 'tool',
      startTime: new Date(record.startedAt),
      input: {
        rawCommand: record.commandRaw,
        canonicalCommand: record.commandCanonical,
        subArgs: record.subArgs,
      },
      metadata: {
        commandKind: record.commandKind,
        aliasApplied: record.aliasApplied || null,
        callerContext: record.callerContext || null,
        cwd: record.cwd,
      } as any,
      tags: [
        'gbrain-command',
        `command:${sanitizeTag(record.commandCanonical)}`,
      ],
    });

    span.update({
      output: {
        exitCode: record.exitCode,
        error: record.error || null,
      },
      metadata: {
        durationMs: record.durationMs,
        finishedAt: record.finishedAt,
      } as any,
    });
    span.end();

    trace.update({
      output: {
        exitCode: record.exitCode,
        error: record.error || null,
      },
      metadata: {
        durationMs: record.durationMs,
      } as any,
    });
    trace.end();

    await client.flush();
  } catch {
    // Telemetry must never break the CLI.
  }
}

export async function withCliCommandTelemetry<T>(
  base: CliCommandTelemetry,
  runner: () => Promise<T>,
): Promise<T> {
  const startedAtMs = Date.now();
  try {
    const result = await runner();
    const exitCode = typeof result === 'number' ? result : 0;
    const record = buildTelemetryRecord(base, {
      exitCode,
      startedAtMs,
      error: null,
    });
    appendTelemetryLog(record);
    await emitOpikCommandSpan(record);
    return result;
  } catch (error) {
    const record = buildTelemetryRecord(base, {
      exitCode: 1,
      startedAtMs,
      error,
    });
    appendTelemetryLog(record);
    await emitOpikCommandSpan(record);
    throw error;
  }
}
