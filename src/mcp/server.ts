import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { ListToolsRequestSchema, CallToolRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import type { BrainEngine } from '../core/engine.ts';
import { operations } from '../core/operations.ts';
import { VERSION } from '../version.ts';
import { buildToolDefs } from './tool-defs.ts';
import { dispatchToolCall, validateParams, buildOperationContext, summarizeMcpParams } from './dispatch.ts';
import { getBrainHotMemoryMeta } from '../core/facts/meta-hook.ts';
import { executeRawJsonb } from '../core/sql-query.ts';

export interface StdioMcpLogEvent {
  operation: string;
  status: 'success' | 'error';
  latencyMs: number;
  toolName?: string;
  params?: unknown;
  errorMessage?: string | null;
  agentName?: string;
}

function stdioAgentName(): string {
  return process.env.GBRAIN_AGENT_NAME || 'stdio';
}

function extractToolErrorMessage(result: { content?: { text?: string }[]; isError?: boolean }): string | null {
  if (!result.isError) return null;
  const text = result.content?.[0]?.text ?? '';
  if (!text) return 'unknown_error';
  try {
    const parsed = JSON.parse(text);
    return parsed.error?.message ?? parsed.message ?? parsed.error ?? 'unknown_error';
  } catch {
    return text.slice(0, 500);
  }
}

export async function logStdioMcpRequest(
  engine: BrainEngine,
  event: StdioMcpLogEvent,
): Promise<void> {
  const paramsSummary = event.toolName
    ? summarizeMcpParams(event.toolName, event.params)
    : null;
  await executeRawJsonb(
    engine,
    `INSERT INTO mcp_request_log (token_name, agent_name, operation, latency_ms, status, error_message, params)
     VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)`,
    [
      'stdio',
      event.agentName || stdioAgentName(),
      event.operation,
      event.latencyMs,
      event.status,
      event.errorMessage || null,
    ],
    [paramsSummary],
  );
}

export async function startMcpServer(engine: BrainEngine) {
  const server = new Server(
    { name: 'gbrain', version: VERSION },
    { capabilities: { tools: {} } },
  );

  // Generate tool definitions from operations. Extracted to buildToolDefs so
  // the subagent tool registry (v0.15+) can call the same mapper against a
  // filtered OPERATIONS subset instead of duplicating this shape.
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    const startedMs = Date.now();
    const response = { tools: buildToolDefs(operations) };
    try {
      await logStdioMcpRequest(engine, {
        operation: 'tools/list',
        status: 'success',
        latencyMs: Date.now() - startedMs,
      });
    } catch { /* best effort */ }
    return response;
  });

  // Dispatch tool calls via shared dispatch.ts (parity with HTTP transport).
  // MCP stdio callers are remote/untrusted; dispatch defaults remote=true.
  // The MCP SDK's response type widened in 1.29 to allow a managed-task wrapper;
  // gbrain ops are synchronous, so we return the legacy `{ content, isError? }`
  // shape and cast through `any` (the SDK accepts it via the ServerResult union).
  server.setRequestHandler(CallToolRequestSchema, async (request: any): Promise<any> => {
    const startedMs = Date.now();
    const { name, arguments: params } = request.params;
    // v0.28: stdio MCP has no per-token auth (local pipe). Default the
    // takes-holder allow-list to ['world'] so agent-facing callers don't
    // see private hunches via takes_list / takes_search / query. Operators
    // who want stdio to see everything should call ops directly via
    // `gbrain call <op>` (sets remote=false in src/cli.ts).
    const result = await dispatchToolCall(engine, name, params, {
      remote: true,
      takesHoldersAllowList: ['world'],
      // v0.31: source defaults to 'default' for stdio (no per-token scope).
      // Operators who want a different source on stdio MCP should set
      // GBRAIN_SOURCE in the env or use --source via `gbrain call`.
      sourceId: process.env.GBRAIN_SOURCE || 'default',
      // v0.31 (eD3): _meta.brain_hot_memory injection so Claude Desktop /
      // Code see the brain's relevant hot memory automatically alongside
      // every tool-call response. Best-effort; absorbs errors.
      metaHook: getBrainHotMemoryMeta,
    });
    try {
      await logStdioMcpRequest(engine, {
        operation: `tools/call:${name}`,
        toolName: name,
        params,
        status: result.isError ? 'error' : 'success',
        errorMessage: extractToolErrorMessage(result),
        latencyMs: Date.now() - startedMs,
      });
    } catch { /* best effort */ }
    return result;
  });

  const transport = new StdioServerTransport();
  await server.connect(transport);

  // Exit cleanly when MCP client disconnects (stdin EOF) or on signals.
  // Without this, orphaned serve processes accumulate and contend for the
  // PGLite write lock, causing ingest jobs (email-sync) to time out.
  let shuttingDown = false;
  const shutdown = (reason: string, code = 0) => {
    if (shuttingDown) return;
    shuttingDown = true;
    process.stderr.write(`[gbrain-serve] shutdown: ${reason}\n`);
    Promise.resolve(engine.disconnect?.())
      .catch(() => {})
      .finally(() => process.exit(code));
  };
  // v0.34.1 (#870): when MCP_STDIO=1, the wrapping gateway (OpenClaw's
  // bundle-mcp layer, others) often pipes the JSON-RPC handshake then
  // closes its stdin half. Treating that as a permanent disconnect kills
  // the server before the first tool call arrives. Signal handlers and
  // transport.onclose still cover the legitimate shutdown paths.
  if (process.env.MCP_STDIO !== '1') {
    process.stdin.on('end', () => shutdown('stdin end'));
    process.stdin.on('close', () => shutdown('stdin close'));
  }
  // @ts-ignore — SDK exposes onclose on transport
  transport.onclose = () => shutdown('transport close');
  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));
  process.on('SIGHUP', () => shutdown('SIGHUP'));
}

// Backward compat: used by `gbrain call` command (trusted local path).
// v0.31.8 (D22): accept opts.sourceId so `gbrain call --source X <op> <json>`
// can scope the op handler to that source. resolveSourceId() in call.ts is
// the upstream resolver; this layer just passes the resolved id through.
export async function handleToolCall(
  engine: BrainEngine,
  tool: string,
  params: Record<string, unknown>,
  opts?: { sourceId?: string },
): Promise<unknown> {
  const op = operations.find(o => o.name === tool);
  if (!op) throw new Error(`Unknown tool: ${tool}`);

  const validationError = validateParams(op, params);
  if (validationError) throw new Error(validationError);

  const ctx = buildOperationContext(engine, params, {
    remote: false,
    logger: { info: console.log, warn: console.warn, error: console.error },
    ...(opts?.sourceId ? { sourceId: opts.sourceId } : {}),
  });

  return op.handler(ctx, params);
}
