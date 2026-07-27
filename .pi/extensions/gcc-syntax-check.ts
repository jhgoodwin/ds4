import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { spawn, spawnSync } from "node:child_process";
import { resolve, isAbsolute, relative, join } from "node:path";
import { existsSync, mkdtempSync, writeFileSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";

const C_SOURCE = /\.c$/i;
const C_HEADER = /\.h$/i;

const MAX_RETRIES = 3;
const SUBAGENT_TIMEOUT_MS = 30_000;

/**
 * Walk NDJSON lines (newest first) for first assistant text block.
 * pi --mode json emits one JSON object per line.
 */
function lastAssistantReply(stdout: string): string {
  for (const line of stdout.trim().split("\n").reverse()) {
    try {
      const event = JSON.parse(line);
      const msg = event.message ?? event;
      if (msg.role === "assistant") {
        const text = (msg.content ?? [])
          .filter((c: { type: string; text?: string }) => c.type === "text")
          .map((c: { text?: string }) => c.text)
          .join("\n");
        if (text?.trim()) return text.trim();
      }
    } catch {
      // malformed NDJSON line — skip
    }
  }
  return "";
}

/**
 * Fork a pi subagent to summarize GCC errors.
 * Writes prompt to temp file, passes via --append-system-prompt.
 */
async function askSubagent(
  gccOutput: string,
  systemPrompt: string,
  signal?: AbortSignal,
): Promise<string> {
  const tmpDir = mkdtempSync(join(tmpdir(), "gcc-syntax-"));
  const promptFile = join(tmpDir, "syntax-analyzer-prompt.md");
  writeFileSync(promptFile, systemPrompt, { mode: 0o600 });

  const task = `Analyze these GCC syntax errors and produce a concise, actionable summary. Do NOT take any action.

\`\`\`
${gccOutput}
\`\`\``;

  try {
    const stdout = await new Promise<string>((resolve, reject) => {
      const proc = spawn("pi", [
        "--mode", "json",
        "-p",
        "--no-session",
        "--append-system-prompt", promptFile,
        task,
      ], {
        stdio: ["ignore", "pipe", "pipe"],
        timeout: SUBAGENT_TIMEOUT_MS,
        signal,
      });

      let out = "";
      let err = "";
      proc.stdout.on("data", (d: Buffer) => { out += d.toString(); });
      proc.stderr.on("data", (d: Buffer) => { err += d.toString(); });
      proc.on("close", (code) => {
        if (code === 0 || out.trim()) resolve(out);
        else reject(new Error(`subagent exit ${code}: ${err.slice(0, 500)}`));
      });
      proc.on("error", reject);
    });

    return lastAssistantReply(stdout) || gccOutput;
  } finally {
    // best-effort temp cleanup (recursive for safety against leftover files)
    try { rmSync(tmpDir, { recursive: true, force: true }); } catch { /* ignore */ }
  }
}

/** Strip YAML frontmatter (--- ... ---) from agent markdown files. */
function stripFrontmatter(content: string): string {
  return content.replace(/^---[\s\S]*?---\n*/, "").trim();
}

export default function (pi: ExtensionAPI) {
  const dirtySources = new Set<string>();
  const cwd = process.cwd();
  let retries = 0;
  let subagentBusy = false;
  let turnNumber = 0;
  let pendingTurn = -1;
  let currentAbort: AbortController | null = null;

  // Load syntax-analyzer system prompt from .pi/agent/agents/
  const agentPromptPath = join(
    process.env.HOME || "/home/jhgoodwin",
    ".pi/agent/agents/syntax-analyzer.md",
  );
  const systemPrompt = existsSync(agentPromptPath)
    ? stripFrontmatter(readFileSync(agentPromptPath, "utf-8"))
    : "Analyze GCC syntax errors. Summarize with file paths, line numbers, and suggested fixes. Do NOT take action.";

  pi.on("tool_result", (event) => {
    if (event.toolName === "edit" || event.toolName === "write") {
      const path = (event.input as { path?: string }).path;
      if (path && (C_SOURCE.test(path) || C_HEADER.test(path))) {
        dirtySources.add(path);
      }
    }
  });

  pi.on("turn_start", () => {
    dirtySources.clear();
    turnNumber++;
    // Kill any subagent still running from a prior turn
    currentAbort?.abort();
    currentAbort = null;
  });

  pi.on("turn_end", () => {
    if (dirtySources.size === 0 || retries >= MAX_RETRIES) return;

    // Only check files that still exist and are within the project
    const sourceFiles = [...dirtySources].filter(f => {
      const abs = isAbsolute(f) ? f : resolve(cwd, f);
      return !relative(cwd, abs).startsWith("..") && existsSync(abs);
    });
    if (sourceFiles.length === 0) return;

    let gccErrors = "";
    try {
      const result = spawnSync("gcc", [
        "-fsyntax-only", "-fmax-errors=10",
        ...sourceFiles.map(f => isAbsolute(f) ? f : resolve(cwd, f)),
      ], { timeout: 15000, encoding: "utf-8", stdio: ["ignore", "pipe", "pipe"] });
      if (result.status !== 0) {
        gccErrors = (result.stderr ?? "").trim();
      }
    } catch (e: unknown) {
      gccErrors = ((e as { stderr?: string }).stderr ?? "").trim();
    }

    if (!gccErrors) {
      retries = 0;
      return;
    }

    retries++;

    // Drop errors silently if prior subagent still running — prevents pile-on
    if (subagentBusy) return;

    // Capture values for closure safety before async boundary
    const capturedErrors = gccErrors;
    const capturedRetries = retries;
    const capturedTurn = turnNumber;
    subagentBusy = true;
    pendingTurn = capturedTurn;

    // Fresh controller allows abort on turn_start / session shutdown
    currentAbort = new AbortController();

    askSubagent(capturedErrors, systemPrompt, currentAbort.signal).then(
      (summary) => {
        subagentBusy = false;
        // Ignore stale result if turn has advanced
        if (pendingTurn !== capturedTurn) return;
        pendingTurn = -1;
        try {
          pi.sendUserMessage(
            [
              "GCC syntax check found errors:",
              "",
              summary,
              capturedRetries < MAX_RETRIES
                ? "\nFix these syntax errors."
                : "\nGiving up after 3 attempts. Manual fix needed.",
            ].join("\n"),
            { deliverAs: "steer" },
          );
        } catch { /* session may have ended */ }
      },
      () => {
        subagentBusy = false;
        if (pendingTurn !== capturedTurn) return;
        pendingTurn = -1;
        try {
          pi.sendUserMessage(
            [
              "GCC syntax check found errors:",
              "",
              capturedErrors,
              capturedRetries < MAX_RETRIES
                ? "\nFix these syntax errors."
                : "\nGiving up after 3 attempts. Manual fix needed.",
            ].join("\n"),
            { deliverAs: "steer" },
          );
        } catch { /* session may have ended */ }
      },
    );
  });
}
