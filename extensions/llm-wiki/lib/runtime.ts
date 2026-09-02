import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { appendFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { loadTaskConfig, noticesEnabled, TASK_DEFAULTS, type TaskConfig } from "./task-config.js";
import { getVaultPaths } from "./utils.js";

/**
 * Background-task runtime for the LLM Wiki (issue #64, part of #63).
 *
 * Provides two primitives, ported from pi-observational-memory's proven
 * pattern, that let the extension perform LLM work WITHOUT blocking the main
 * agent turn:
 *
 *   - launchTask(): fire-and-forget a detached promise that may outlive the
 *     current turn. The in-flight promise is stored so callers can await it at
 *     compaction / session exit (so background work is never silently lost),
 *     but the agent loop itself never blocks on it. Single-flight per label
 *     to avoid pile-ups.
 *
 *   - resolveModel(): pick the model for background work — configured
 *     `taskModel` → session model fallback → API-key resolution. Returns a
 *     discriminated result so callers degrade gracefully (keep the existing
 *     synchronous main-agent flow) when no model / API key is available.
 *
 * This module introduces NO user-facing behavior on its own; it is the
 * infrastructure that issues #65 (background ingest), #66 (background
 * embeddings) and #69 (model selection) build upon.
 */

export type ResolveResult =
  | { ok: true; model: unknown; apiKey: string; headers?: Record<string, string> }
  | { ok: false; reason: string };

type NotifyLevel = "info" | "warning" | "error";
type Notify = (message: string, type?: NotifyLevel) => void;

export interface ResolveCtx {
  /** Current session model (may be undefined when the session has no model). */
  model: unknown;
  modelRegistry: {
    find(provider: string, id: string): unknown;
    getApiKeyAndHeaders(
      model: unknown,
    ): Promise<{ ok: boolean; apiKey?: string; headers?: Record<string, string> }>;
  };
  hasUI: boolean;
  ui?: { notify: Notify };
}

export interface LaunchCtx {
  hasUI: boolean;
  ui?: { notify: Notify };
}

export class Runtime {
  config: TaskConfig = { ...TASK_DEFAULTS };

  /**
   * Extension API handle, attached at registration. Used by `report()` to emit
   * visible completion messages for background actions (issue #77). Optional so
   * the Runtime stays unit-testable without a live `pi`.
   */
  pi?: ExtensionAPI;

  /** Labels of tasks currently in flight (single-flight guard per label). */
  private inFlightLabels = new Set<string>();
  /** All in-flight task promises, keyed for await-at-exit and dedupe. */
  private inFlight = new Map<string, Promise<void>>();
  /** Whether we've already surfaced a model-resolution failure (avoid spam). */
  resolveFailureNotified = false;

  private cwd: string | undefined;

  ensureConfig(cwd: string): void {
    this.cwd = cwd;
    this.config = loadTaskConfig(cwd);
  }

  /** True if a task with this label is currently running. */
  isInFlight(label: string): boolean {
    return this.inFlightLabels.has(label);
  }

  /** Number of background tasks currently running. */
  get pendingCount(): number {
    return this.inFlight.size;
  }

  /**
   * Resolve the model + auth for background work.
   *
   * Precedence (issue #69): per-call `override` → configured `taskModel` →
   * session model. Each configured layer is applied only when the model is
   * found in the registry; a missing layer warns (when UI is available) and
   * falls through to the next. Returns { ok: false } when nothing resolves or
   * no API key exists, so callers can fall back to the synchronous
   * main-agent path.
   */
  async resolveModel(
    ctx: ResolveCtx,
    override?: { provider: string; id: string },
  ): Promise<ResolveResult> {
    let model = ctx.model;

    // Configured taskModel layer (beats the session model).
    const configured = this.config.taskModel;
    if (configured) {
      const found = ctx.modelRegistry.find(configured.provider, configured.id);
      if (found) {
        model = found;
      } else if (ctx.hasUI && ctx.ui) {
        ctx.ui.notify(
          `LLM Wiki: configured task model ${configured.provider}/${configured.id} not found, using session model`,
          "warning",
        );
      }
    }

    // Per-call override layer (beats both config and session).
    if (override) {
      const found = ctx.modelRegistry.find(override.provider, override.id);
      if (found) {
        model = found;
      } else if (ctx.hasUI && ctx.ui) {
        ctx.ui.notify(
          `LLM Wiki: model override ${override.provider}/${override.id} not found, using ${
            configured ? "configured/session" : "session"
          } model`,
          "warning",
        );
      }
    }

    if (!model) {
      return {
        ok: false,
        reason: "no model available (session has no model and no taskModel configured)",
      };
    }

    const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
    if (!auth.ok) {
      const provider = (model as { provider?: string }).provider ?? "unknown";
      return { ok: false, reason: `no API key for provider "${provider}"` };
    }
    return { ok: true, model, apiKey: auth.apiKey ?? "", headers: auth.headers };
  }

  /**
   * Fire-and-forget a background task.
   *
   * The work runs in a detached promise so the caller (an agent hook/tool)
   * is never blocked. Errors are caught and surfaced via the UI (when
   * available) instead of crashing the agent. Single-flight per label: if a
   * task with the same label is already running, the new request is dropped
   * and the existing promise is returned.
   *
   * The returned promise resolves when the work completes; hold onto it (or
   * call awaitAll) to drain background work before compaction/exit.
   */
  launchTask(ctx: LaunchCtx, label: string, work: () => Promise<void>): Promise<void> {
    const existing = this.inFlight.get(label);
    if (existing) {
      // Duplicate launch: make the drop visible instead of a silent no-op.
      try {
        if (ctx.hasUI && ctx.ui)
          ctx.ui.notify(`LLM Wiki: ${label} 已在运行，跳过重复任务`, "info");
      } catch {
        // stale ctx must not propagate
      }
      return existing;
    }

    // Capture ctx properties synchronously — after `await work()` the extension
    // ctx may be stale (e.g. after newSession/fork/switchSession/reload), and
    // accessing ctx.hasUI or ctx.ui on a stale proxy throws.
    const hasUI = ctx.hasUI;
    const ui = ctx.ui;

    this.inFlightLabels.add(label);
    // biome-ignore lint/style/useConst: referenced inside its own initializer (finally block)
    let promise!: Promise<void>;
    promise = (async () => {
      try {
        await work();
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        try {
          if (hasUI && ui) ui.notify(`LLM Wiki: ${label} failed: ${msg}`, "warning");
        } catch {
          console.warn(`[llm-wiki] ${label} failed: ${msg} (notify unavailable)`);
        }
      } finally {
        this.inFlightLabels.delete(label);
        if (this.inFlight.get(label) === promise) this.inFlight.delete(label);
      }
    })();
    this.inFlight.set(label, promise);
    return promise;
  }

  /**
   * Report a completed background action to the user (issue #77).
   *
   * Every mutating wiki action runs off the agent's critical path; this is how
   * the work becomes visible. Emits a `wiki-action-report` custom message,
   * shown in the UI when notices are enabled (the `notices` config, default
   * on) and otherwise injected silently. Delivered as `nextTurn` so it never
   * interrupts or triggers a turn. Never throws — reporting must not crash the
   * background task that called it.
   */
  report(summary: string, opts?: { display?: boolean; ui?: { notify: Notify } }): void {
    if (!summary) return;
    const display = opts?.display ?? noticesEnabled(this.config);
    // Durable fallback: persist every report so a lost nextTurn delivery is
    // still recoverable instead of vanishing without a trace.
    try {
      const paths = getVaultPaths(this.cwd ?? process.cwd());
      mkdirSync(paths.meta, { recursive: true });
      appendFileSync(
        join(paths.meta, "background-reports.md"),
        `## ${new Date().toISOString()}\n\n${summary}\n\n`,
        "utf8",
      );
    } catch (err) {
      console.warn("[llm-wiki] report persistence failed:", err);
    }
    if (!this.pi) return;
    try {
      this.pi.sendMessage(
        { customType: "wiki-action-report", content: summary, display },
        { deliverAs: "nextTurn" },
      );
    } catch (err) {
      // Reporting is best-effort; the persisted copy above is the fallback.
      console.warn("[llm-wiki] report delivery failed:", err);
      try {
        opts?.ui?.notify("LLM Wiki: 后台报告已写入 meta/background-reports.md", "info");
      } catch {
        // best-effort
      }
    }
  }

  /**
   * Run a mutating action in the background and report its result (issue #77).
   *
   * Thin wrapper over `launchTask`: `work` performs the off-thread mutation and
   * returns a one-line human summary (or null to stay silent). On success the
   * summary is surfaced via `report()`. Single-flight, error-isolated, and
   * awaited-at-exit exactly like `launchTask`.
   */
  launchReported(ctx: LaunchCtx, label: string, work: () => Promise<string | null>): Promise<void> {
    return this.launchTask(ctx, label, async () => {
      // Capture synchronously — after `await work()` the extension ctx may be
      // a stale proxy (newSession/fork/switchSession/reload) and accessing
      // ctx.hasUI or ctx.ui on it throws (see launchTask).
      const hasUI = ctx.hasUI;
      const ui = ctx.ui;
      const summary = await work();
      if (summary) {
        // Primary channel first: a stale-UI notify must never skip the report.
        this.report(summary, { ui });
        // Instant completion feedback (best-effort toast — a stale UI handle
        // must not throw and take down the task).
        try {
          if (hasUI && ui) ui.notify(summary.split("\n")[0].replace(/\*\*/g, ""), "info");
        } catch {
          // best-effort
        }
      }
    });
  }

  /**
   * Await all in-flight background tasks. Call at compaction / session exit so
   * background work is not lost. Never rejects — task errors are already
   * isolated inside launchTask.
   */
  async awaitAll(): Promise<void> {
    while (this.inFlight.size > 0) {
      await Promise.allSettled([...this.inFlight.values()]);
    }
  }
}

/**
 * Register the shared background runtime and wire it into the extension
 * lifecycle: config is loaded lazily per turn, and in-flight tasks are drained
 * before compaction and on shutdown so background work is never lost.
 *
 * Returns the Runtime instance so concrete background workers (issues #65,
 * #66) can launch tasks on it.
 */
export function registerBackgroundRuntime(pi: ExtensionAPI): Runtime {
  const runtime = new Runtime();
  // Attach the API so background tasks can emit visible completion reports
  // (issue #77). Done here (not in the constructor) to keep Runtime testable.
  runtime.pi = pi;

  pi.on("turn_start", (_event, ctx) => {
    runtime.ensureConfig(ctx.cwd);
  });

  // Drain in-flight background work before the session is compacted or shut
  // down, so nothing is lost mid-flight.
  pi.on("session_before_compact", async () => {
    await runtime.awaitAll();
  });
  pi.on("session_shutdown", async () => {
    await runtime.awaitAll();
  });

  return runtime;
}
