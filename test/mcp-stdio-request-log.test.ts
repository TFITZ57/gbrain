import { describe, expect, test } from 'bun:test';
import { logStdioMcpRequest } from '../src/mcp/server.ts';

describe('stdio MCP request logging', () => {
  test('writes redacted params with stdio token and agent name', async () => {
    const calls: Array<{ sql: string; params: unknown[] }> = [];
    const engine = {
      executeRaw: async (sql: string, params: unknown[]) => {
        calls.push({ sql, params });
        return [];
      },
    } as any;

    await logStdioMcpRequest(engine, {
      operation: 'tools/call:put_page',
      toolName: 'put_page',
      params: {
        slug: 'people/private-name',
        content: 'sensitive body',
        'leak-key-private-name': 'ignored',
      },
      status: 'success',
      latencyMs: 17,
      agentName: 'hermes',
    });

    expect(calls).toHaveLength(1);
    expect(calls[0].sql).toContain('mcp_request_log');
    expect(calls[0].params[0]).toBe('stdio');
    expect(calls[0].params[1]).toBe('hermes');
    expect(calls[0].params[2]).toBe('tools/call:put_page');
    expect(calls[0].params[4]).toBe('success');
    const summary = calls[0].params[6] as Record<string, unknown>;
    expect(summary.redacted).toBe(true);
    expect(summary.declared_keys).toEqual(['content', 'slug']);
    expect(summary.unknown_key_count).toBe(1);
    expect(JSON.stringify(summary)).not.toContain('private-name');
    expect(JSON.stringify(summary)).not.toContain('sensitive body');
    expect(JSON.stringify(summary)).not.toContain('leak-key');
  });

  test('records safe error message for failed calls', async () => {
    const calls: Array<{ sql: string; params: unknown[] }> = [];
    const engine = {
      executeRaw: async (sql: string, params: unknown[]) => {
        calls.push({ sql, params });
        return [];
      },
    } as any;

    await logStdioMcpRequest(engine, {
      operation: 'tools/call:nope',
      toolName: 'nope',
      params: {},
      status: 'error',
      errorMessage: 'Unknown tool: nope',
      latencyMs: 3,
    });

    expect(calls[0].params[0]).toBe('stdio');
    expect(calls[0].params[1]).toBe('stdio');
    expect(calls[0].params[4]).toBe('error');
    expect(calls[0].params[5]).toBe('Unknown tool: nope');
  });
});
