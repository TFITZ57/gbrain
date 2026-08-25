/**
 * test/resolve-ipc-binding.test.ts — #4474.
 *
 * `gbrain serve --http` never bound the resolve-IPC unix socket (the
 * listener lived inline in the stdio MCP path only), so on the exact
 * posture `gbrain bootstrap harness` targets, every wired lifecycle hook
 * degraded to `no_serve` forever — with no local recovery on PGLite (the
 * http serve owns the single-writer lock). The wiring now lives in the
 * shared `bindResolveIpcForServe` helper and BOTH transports call it.
 */

import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import net from "node:net";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  bindResolveIpcForServe,
  startResolveIpcSupervisorForServe,
  warmResolveIpcPool,
  type ResolveIpcBinding,
} from "../src/mcp/resolve-ipc-binding.ts";
import { resolveSocketPath } from "../src/core/context/resolve-ipc.ts";
import type { BrainEngine } from "../src/core/engine.ts";
import { withEnv } from "./helpers/with-env.ts";

const REPO_ROOT = join(import.meta.dir, "..");
const readSrc = (rel: string) => Bun.file(join(REPO_ROOT, rel));

let tmp: string;

async function waitFor(
  predicate: () => boolean,
  timeoutMs = 500,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error("condition_timeout");
    await Bun.sleep(5);
  }
}

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "gb-ipc-bind-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

describe("bindResolveIpcForServe (#4474)", () => {
  it("binds the socket for a PGLite config and close() reaps it", async () => {
    const dataDir = join(tmp, "db");
    mkdirSync(dataDir, { recursive: true });
    mkdirSync(join(tmp, ".gbrain"), { recursive: true });
    writeFileSync(
      join(tmp, ".gbrain", "config.json"),
      JSON.stringify({ engine: "pglite", database_path: dataDir }),
    );
    await withEnv(
      {
        GBRAIN_HOME: tmp,
        GBRAIN_DATABASE_URL: undefined,
        DATABASE_URL: undefined,
      },
      async () => {
        // Bind-time never touches the engine (handlers close over it lazily),
        // so a stub is enough to prove the listener itself comes up.
        const binding = await bindResolveIpcForServe(
          {} as unknown as BrainEngine,
          "default",
        );
        try {
          expect(binding.server).not.toBeNull();
          expect(binding.socketPath).toBe(resolveSocketPath(dataDir));
          expect(existsSync(binding.socketPath!)).toBe(true);
        } finally {
          await binding.close();
        }
        expect(existsSync(resolveSocketPath(dataDir))).toBe(false);
        // close() is idempotent.
        await binding.close();
      },
    );
  });

  it("returns a null binding (not a throw) when the config has no keying material", async () => {
    mkdirSync(join(tmp, ".gbrain"), { recursive: true });
    writeFileSync(
      join(tmp, ".gbrain", "config.json"),
      JSON.stringify({ engine: "pglite" }),
    );
    await withEnv(
      {
        GBRAIN_HOME: tmp,
        GBRAIN_DATABASE_URL: undefined,
        DATABASE_URL: undefined,
      },
      async () => {
        const binding = await bindResolveIpcForServe(
          {} as unknown as BrainEngine,
          "default",
        );
        expect(binding.server).toBeNull();
        expect(binding.socketPath).toBeNull();
        await binding.close(); // no-op, must not throw
      },
    );
  });

  it("close destroys an idle accepted client and remains bounded", async () => {
    const dataDir = join(tmp, "db");
    mkdirSync(dataDir, { recursive: true });
    mkdirSync(join(tmp, ".gbrain"), { recursive: true });
    writeFileSync(
      join(tmp, ".gbrain", "config.json"),
      JSON.stringify({ engine: "pglite", database_path: dataDir }),
    );
    await withEnv(
      {
        GBRAIN_HOME: tmp,
        GBRAIN_DATABASE_URL: undefined,
        DATABASE_URL: undefined,
      },
      async () => {
        const binding = await bindResolveIpcForServe(
          {} as unknown as BrainEngine,
          "default",
        );
        expect(binding.socketPath).not.toBeNull();
        const client = net.createConnection(binding.socketPath!);
        await new Promise<void>((resolve, reject) => {
          client.once("connect", resolve);
          client.once("error", reject);
        });
        const clientClosed = new Promise<void>((resolve) =>
          client.once("close", () => resolve()),
        );
        const started = Date.now();
        await binding.close();
        expect(Date.now() - started).toBeLessThan(500);
        await Promise.race([
          clientClosed,
          Bun.sleep(250).then(() => {
            throw new Error("client_close_timeout");
          }),
        ]);
        expect(client.destroyed).toBe(true);
      },
    );
  });
});

describe("durable resolve IPC owner", () => {
  it("warms both concurrent turn-context database lanes", async () => {
    let calls = 0;
    let active = 0;
    let maxActive = 0;
    const releases: Array<() => void> = [];
    const engine = {
      executeRaw: () =>
        new Promise<unknown[]>((resolve) => {
          calls += 1;
          active += 1;
          maxActive = Math.max(maxActive, active);
          releases.push(() => {
            active -= 1;
            resolve([]);
          });
        }),
    } as unknown as BrainEngine;

    const work = warmResolveIpcPool(engine, new AbortController().signal);
    await waitFor(() => calls === 2);
    expect(maxActive).toBe(2);
    for (const release of releases) release();
    await work;
    expect(calls).toBe(2);
  });

  it("retries an initial contender loss and reacquires a lost owner", async () => {
    let attempts = 0;
    let firstListening = true;
    let firstCloses = 0;
    let secondCloses = 0;
    const binding = (
      socketPath: string,
      listening: () => boolean,
      close: () => void,
    ): ResolveIpcBinding => ({
      server: {} as ResolveIpcBinding["server"],
      socketPath,
      isListening: listening,
      close: async () => {
        close();
      },
    });
    const first = binding(
      "/tmp/first.sock",
      () => firstListening,
      () => {
        firstCloses += 1;
      },
    );
    const second = binding(
      "/tmp/second.sock",
      () => true,
      () => {
        secondCloses += 1;
      },
    );

    const supervisor = await startResolveIpcSupervisorForServe(
      {} as BrainEngine,
      "default",
      {
        retryMs: 10,
        monitorMs: 10,
        keepalive: null,
        starter: async () => {
          attempts += 1;
          if (attempts === 1) return null;
          return attempts === 2 ? first : second;
        },
      },
    );
    await waitFor(() => attempts === 2);
    firstListening = false;
    await waitFor(() => attempts === 3);
    expect(firstCloses).toBe(1);
    await supervisor.close();
    expect(secondCloses).toBe(1);
  });

  it("keeps an owned Postgres connection warm on the configured interval", async () => {
    let pings = 0;
    const owner: ResolveIpcBinding = {
      server: {} as ResolveIpcBinding["server"],
      socketPath: "/tmp/owner.sock",
      isListening: () => true,
      close: async () => {},
    };
    const supervisor = await startResolveIpcSupervisorForServe(
      {} as BrainEngine,
      "default",
      {
        monitorMs: 10,
        keepaliveMs: 10,
        keepalive: async () => {
          pings += 1;
        },
        starter: async () => owner,
      },
    );
    await waitFor(() => pings >= 2);
    await supervisor.close();
    expect(pings).toBeGreaterThanOrEqual(2);
  });

  it("a wedged keepalive cannot freeze ownership monitoring, reacquisition, or close", async () => {
    let attempts = 0;
    let firstListening = true;
    let pings = 0;
    let aborted = false;
    let firstCloses = 0;
    let secondCloses = 0;
    const binding = (
      listening: () => boolean,
      close: () => void,
    ): ResolveIpcBinding => ({
      server: {} as ResolveIpcBinding["server"],
      socketPath: "/tmp/owner.sock",
      isListening: listening,
      close: async () => {
        close();
      },
    });
    const first = binding(
      () => firstListening,
      () => {
        firstCloses += 1;
      },
    );
    const second = binding(
      () => true,
      () => {
        secondCloses += 1;
      },
    );
    const supervisor = await startResolveIpcSupervisorForServe(
      {} as BrainEngine,
      "default",
      {
        monitorMs: 10,
        retryMs: 10,
        keepaliveMs: 10,
        keepaliveTimeoutMs: 20,
        keepalive: async (signal) => {
          pings += 1;
          signal.addEventListener(
            "abort",
            () => {
              aborted = true;
            },
            { once: true },
          );
          return new Promise<never>(() => {});
        },
        starter: async () => {
          attempts += 1;
          return attempts === 1 ? first : second;
        },
      },
    );
    await waitFor(() => pings === 1);
    firstListening = false;
    await waitFor(() => attempts === 2);
    expect(firstCloses).toBe(1);
    await waitFor(() => aborted);
    const started = Date.now();
    await supervisor.close();
    expect(Date.now() - started).toBeLessThan(500);
    expect(secondCloses).toBe(1);
  });
});

describe("both serve transports bind through the shared helper (#4474)", () => {
  it("serve --http wires the durable supervisor with teardown", async () => {
    const src = await readSrc("src/commands/serve-http.ts").text();
    expect(src).toContain("startResolveIpcSupervisorForServe(");
    expect(src).toContain("await ipcSupervisor.close()");
  });

  it("the stdio MCP path wires bindResolveIpcForServe with teardown", async () => {
    const src = await readSrc("src/mcp/server.ts").text();
    expect(src).toContain("bindResolveIpcForServe(");
    expect(src).toContain("ipcBinding.close()");
  });

  it("bootstrap verify prefers a live serve socket over self-creating one", async () => {
    // verify.ts:hooks smoke used to ALWAYS start its own IPC server, which
    // manufactured the condition under test and masked serve postures that
    // never bind IPC. Pin the live-socket branch.
    const src = await readSrc("src/core/bootstrap/verify.ts").text();
    expect(src).toContain("const liveSocket = existsSync(socketPath)");
    expect(src).toMatch(/if \(!liveSocket\) \{/);
  });
});
