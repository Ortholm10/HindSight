# CLAUDE.md — Hindsight

## 1. Project Context

**Hindsight is an agent that audits a backtest or time-series pipeline for look-ahead bias, and refuses to report any finding it has not proven by execution.**

The bug class: code accidentally reads data that would not have existed at decision time. `df['close'].rolling(20).mean()` includes today's close, but the trade enters at today's open. The code never crashes, never fails a test — it just produces a better number. Reported Sharpe 4.12; actual Sharpe 0.34.

**The core mechanic.** The agent never invents or predicts values. `.shift(1)` slides a column down one row so each row carries the previous *completed* value. Nothing is created — data the user was never entitled to is removed, and the strategy is re-run.

**Where the agent is.** Not in "should I shift this line" — that is one LLM call. The agent lives in *"is this number believable yet, and what do I do next to find out?"* Three places the straight line breaks:

1. The fix is not always one line. Higher-timeframe merge bugs need a repair constructed, run, its traceback read, and retried.
2. Fixing one leak reveals another. 4.12 → 2.81 → 1.4 → 0.34. The agent must judge that 2.81 is *still* implausible for a daily crossover strategy and keep going. Loop length is unknown in advance.
3. Tests are not always conclusive. The agent must distinguish "leak proven" from "my patch was wrong" from "no trades on this data."

**Users.** Anyone with a Python file or notebook, pandas, free data, and a suspiciously good result. Framed in the README as *leak auditing for any time-series pipeline, demonstrated on backtests* — the same bug class appears in demand forecasting, predictive maintenance, credit risk, and medical prognosis.

**This is a hackathon build.** Judged on: Problem & User Value (15), Agent Solution & Engineering (30), End-to-End Quality (20), Measured Improvement (15), Reproducibility (15), Hot Take (5). Every scoping decision below traces back to one of those.

---

## 2. Architecture

```
hindsight/
├── hindsight_core/           ← ALL logic lives here
│   ├── orchestrator.py       agent loop: state → decide → tool → observe → update
│   ├── tools/
│   │   ├── scan_file.py      AST/libcst scan → candidate leak sites
│   │   ├── read_context.py   pull surrounding code for a candidate
│   │   ├── apply_patch.py    libcst transform, returns diff
│   │   ├── run_backtest.py   sandboxed subprocess, returns metrics + run_id
│   │   ├── compare_runs.py   before/after metric delta
│   │   └── check_memory.py   query confirmed leak signatures
│   ├── provers/              subagents, own retry loops, run in parallel
│   ├── hooks/
│   │   └── verification.py   BLOCKS any finding lacking an execution record
│   ├── memory.py             JSON store of confirmed leak signatures
│   ├── sandbox.py            subprocess + timeout + resource caps
│   ├── events.py             typed event emitter
│   └── models.py             LLM client wrappers, fallback chain
├── hindsight_cli/            thin wrapper → hindsight_core
├── hindsight_server/         FastAPI + SSE, thin wrapper → hindsight_core
├── web/                      Vite + React frontend
├── eval/
│   ├── cases/                20 frozen cases: 12 injected, 8 clean
│   ├── inject.py             leak injection from the taxonomy
│   ├── baselines/            one-shot LLM prompt; freqtrade lookahead-analysis
│   └── harness.py            runs all cases, emits results table
├── .claude/skills/           project skills — one per leak type
└── docs/
    ├── taxonomy.md           leak types: detect / fix / inject
    ├── CHANGELOG.md          the Improvement Changelog
    └── REPRODUCE.md          exact commands from a clean environment
```

**The one rule that governs the whole layout:** the web server and the CLI call the *exact same* functions in `hindsight_core`. No logic is duplicated across the two entry points, ever. If a behaviour exists in the web path but not the CLI path, it is a bug.

**Event stream schema** (emitted by core, consumed by both CLI printer and SSE): `scan_complete`, `triage`, `baseline`, `prove_start`, `prove_result`, `agent_decision`, `final`.

**Where things belong:**

| If you are adding… | It goes in… |
|---|---|
| A new leak type | `docs/taxonomy.md` + `.claude/skills/` + `eval/inject.py` |
| A new agent capability | `hindsight_core/tools/` — and it must be registered with the orchestrator |
| Anything that touches the LLM | `hindsight_core/models.py` — nowhere else |
| A display concern | `hindsight_cli/` or `web/` — never in core |
| A new metric | `hindsight_core/tools/compare_runs.py` |

---

## 3. Code Style

- **Python 3.11+.** Type hints on every public function signature. `ruff` for lint and format; default line length 88.
- **Dataclasses over dicts** for anything crossing a module boundary. Events, findings, and run records are all typed dataclasses in `models.py` — not free-form dicts.
- **Every finding carries a `run_id`.** A `Finding` without `before_run_id` and `after_run_id` is structurally invalid and cannot be constructed. Enforce this in `__post_init__`, not in a review comment.
- **No bare `except`.** Sandbox failures, timeouts, and zero-trade runs are *distinct outcomes* with distinct handling — never collapse them into one error path.
- **Functions do one thing and return data.** Tools return structured results; the orchestrator decides. A tool never decides whether to continue the loop.
- **Prints go through the event emitter.** No `print()` in `hindsight_core`. The CLI subscribes to events and prints; core stays silent.
- **Docstrings only where the *why* is non-obvious.** Skip restating the signature. Do explain any line where a shift, offset, or index alignment is deliberate — those are the lines a reader will get wrong.
- **Frontend:** functional components, hooks, Tailwind utility classes inline. No CSS modules, no styled-components. Numbers render in a monospace face so digits align in columns.

**Naming:** `leak_*` for detection, `prove_*` for execution-backed verification, `run_*` for sandbox execution. Never use "detect" and "prove" interchangeably in code or output — the distinction between them is the entire product.

---

## 4. Preferred Libraries

**Use these. Do not substitute without asking.**

| Purpose | Library | Why it and not the alternative |
|---|---|---|
| Code parsing | `ast` (stdlib) | Read-only scanning — fast, zero deps |
| Code rewriting | `libcst` | Preserves formatting and comments; `ast` cannot round-trip |
| Backtest engine | `backtesting.py` | MIT, small, deterministic, no broker setup |
| Market data | `yfinance` | Free, no key, cacheable to disk |
| Dataframes | `pandas` | Non-negotiable — it is the substrate the bugs live in |
| Server | `FastAPI` + `sse-starlette` | SSE without websocket complexity |
| Tests | `pytest` | Plus `pytest-timeout` for sandbox tests |
| Frontend build | **Vite** | Not Next.js — no SSR needed, faster cold start, simpler repro |
| Charts | `recharts` | Equity-curve overlay is the key visual |
| Animation | `framer-motion` | Sparingly — streaming events only |

**LLM providers** (all free tier): Gemini Flash for volume, OpenRouter as fallback and as baseline model #2, Groq only for small fast classification (its TPM ceiling is too low for whole files). All access goes through `models.py` with an explicit fallback chain. Verify live quota limits before relying on a number.

**No model training. No fine-tuning. No dataset.** All leak knowledge lives in editable text files under `.claude/skills/` and `docs/taxonomy.md`.

**Do not add:** LangChain, LlamaIndex, CrewAI, or any agent framework. The orchestrator loop is ~150 lines of Python and being able to show that loop is a scoring asset. Also no database, no ORM, no auth library — see Critical Rules.

---

## 5. Skills & Plugins

### 5.1 What is installed, and what ships

**Superpowers is a development-environment tool. It is never a runtime dependency of Hindsight.**

This is a hard boundary and it protects 15 reproducibility points. A judge clones the repo on a clean machine and runs the commands in `REPRODUCE.md`. If any of those commands need Superpowers, ECC, a Claude Code plugin, or a `.claude/` directory to be present, the run fails and those points are gone. `requirements.txt` and `package.json` must be sufficient. The `.claude/skills/` directory in this repo holds *Hindsight's own* leak-type knowledge files, which are read by `hindsight_core` as plain text — they have nothing to do with Claude Code plugins.

ECC **is installed** alongside Superpowers, and must be used narrowly. Both frameworks want to own the workflow, so the division is fixed:

- **Superpowers owns the process.** `using-superpowers` is the dispatcher. Plan → TDD → verify → finish runs through it.
- **ECC is a library, not a driver.** Invoke exactly four of its skills, by name, only where the phase table says so: `eval-harness`, `verification-loop`, `python-testing`, `mle-workflow`. Never let it route the session.
- **Do not use ECC's `multi-*` commands.** They need the separate `ccg-workflow` runtime, which is not installed and is not worth installing this weekend.
- **Keep ECC rules limited to `common` and `python`.** Rules are always-loaded context; the rest are dead weight here.
- **Skip ECC's memory vault, instincts, and continuous learning.** Hindsight has its own memory layer and does not need a second one.

Live risk to watch: ECC ships ~286 skills into `~/.claude/skills/`, and this repo's own leak-type skills live in `.claude/skills/`. If a session starts reaching for unrelated ECC skills, or if Superpowers' chain stops firing, that is the collision — say "check your skills" first, and disable ECC for the session if it persists. Neither framework ever ships with Hindsight.

### 5.2 The four Superpowers skills that matter here

**`verification-before-completion` — always on, every phase, no exceptions.**

This is the single most important skill for this project, because it is Hindsight's own thesis pointed back at the agent building it. It prohibits claiming completion without running the actual command and reading its output, and bans the words "should work" and "probably."

The specific failure it prevents: Claude writes `prove_leak()`, reasons convincingly about why it is correct, and reports it working — without ever having executed a backtest. That failure is *especially* likely in this codebase, because the whole domain is bugs that produce plausible output instead of crashing. A prover that silently returns "no leak found" for every case looks exactly like a prover that works. If we are building a tool whose entire pitch is "an agent's confidence is worthless on silent bugs," we cannot let the agent building it run on confidence.

**`test-driven-development` — scoped, not global.**

Mandatory for:
- `eval/` — the harness and the 20 cases
- `hindsight_core/provers/` — the differential execution prover
- `hindsight_core/sandbox.py` — timeout, crash, and zero-trade paths
- `hindsight_core/hooks/verification.py` — the blocking hook

Waived for:
- `web/` — all React components
- `hindsight_cli/` — output formatting
- One-off scripts

Rationale for the split: in the core, a wrong result is *silent* — a broken prover reports "clean" and nothing looks wrong. In the UI, a wrong result is visible the moment you open the page, so the test adds cost without adding information. This override is legitimate; priority order is your instructions, then Superpowers skills, then default behaviour.

**`writing-plans` — before each build session.**

Produces plans where each task is 2–5 minutes with exact file paths and complete code blocks, saved to `docs/superpowers/plans/`. This matters for a specific reason: detailed plans are what make parallel terminals possible. Two sessions can only run side by side without collision if each has a plan precise enough that neither needs to touch the other's files.

**`dispatching-parallel-agents` + `using-git-worktrees` — the multi-session phases.**

Worktrees give each parallel session an isolated workspace so nothing lands on `main` mid-build. Use for the genuinely independent work only — see the phase table.

### 5.3 Skills deliberately skipped

**`brainstorming` — skip it.** It adds 10–20 minutes of design questions before code. Across ~7 sessions that is up to 2 hours spent re-deciding questions this project has already settled. Every session prompt in `docs/sessions.md` hands over a settled design; go straight to `writing-plans` with that spec. If a session genuinely opens a new design question, run it — but the default is skip.

**`subagent-driven-development` — skip; use `executing-plans`.** With ~30 working hours you want to watch the work land, not autonomously dispatch and audit afterwards. Choose `executing-plans` at the fork.

**`systematic-debugging` — do not invoke it manually.** It auto-triggers on failure and that is the correct behaviour. Worth knowing it questions the architecture after 3 failed fixes; if it does that in the sandbox, listen.

**`requesting-code-review` / `receiving-code-review` — only once**, before the final commit on Monday. Per-task review is a luxury this timeline cannot afford.

**`finishing-a-development-branch`** — at the end of each worktree phase. It gates on tests, which is what you want before merging into `main`.

### 5.4 Skill map by phase

| # | Phase | Skills to invoke | Model / effort | Done means |
|---|---|---|---|---|
| 1 | Eval harness + 20 cases + baselines | `writing-plans` → `test-driven-development` → `verification-before-completion`; read ECC `eval-harness` | Opus, high | `hindsight eval --suite all` runs, both baselines recorded to JSON |
| 2 | CLI straight-line pipeline | `writing-plans` → TDD (core only) → verification | Opus, high | scan → triage → prove runs end to end on one case |
| 3 | Agent loop + provers + hook | `writing-plans` → TDD → `dispatching-parallel-agents` → verification; read ECC `verification-loop` | **Opus, maximum** | Agent finds a second leak after fixing the first, unprompted |
| 4 | Event stream + FastAPI + SSE | `writing-plans` → verification (TDD waived) | Sonnet, medium | Events stream to browser in real time |
| 5 | Web UI, 3 screens | `writing-plans` → verification (TDD waived) | Sonnet, medium | Drop file → live stream → verdict with equity curves |
| 6 | README, REPRODUCE, CHANGELOG | verification | Sonnet, low | Commands copy-pasted into a clean container and pass |
| 7 | Video | — | — | Under 5:00, one unedited agent run |

Phase 3 is the one that earns the 30 Agent Solution points. Give it the most capable model and the most patience. Phases 4 and 5 are mechanical — do not spend Opus tokens there.

**Parallelise only phases 4 and 5** (server and frontend, in separate worktrees, against the frozen event schema). Phases 1–3 are strictly sequential — each one's output is the next one's input, and running them in parallel produces merge conflicts in `hindsight_core` that will cost more than they save.

**Order note:** phases 2 and 3 are deliberately separate. Building the straight-line pipeline *first* and the agent loop *second* produces the pipeline-versus-agent delta, which is the headline entry in the Improvement Changelog. Do not collapse them.

---

## 6. Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add GEMINI_API_KEY / OPENROUTER_API_KEY

# Audit a single file
hindsight audit path/to/strategy.py
hindsight audit path/to/strategy.py --json --out report.json
hindsight audit path/to/notebook.ipynb        # if notebook support ships

# Evaluation
hindsight eval --suite all                    # all 20 cases
hindsight eval --suite injected               # the 12 with known leaks
hindsight eval --suite clean                  # the 8 controls — FP check
hindsight eval --case htf_merge_11346         # the hard case
hindsight eval --baseline oneshot             # baseline #1
hindsight eval --baseline freqtrade           # baseline #2

# Tests
pytest                                        # everything
pytest tests/core -x                          # core, stop on first failure
pytest tests/provers --timeout=120            # sandbox tests need headroom
ruff check . && ruff format .

# Servers
uvicorn hindsight_server.main:app --reload --port 8000
cd web && npm run dev                         # port 5173

# Data
python -m eval.cache_data                     # pre-cache 1y daily bars
```

Every command above must run from a clean clone with nothing but `requirements.txt` and `npm install`. If one of them stops being true, `REPRODUCE.md` is broken and 15 points are at risk.

---

## 7. Critical Rules

**1. No finding reaches the report without an attached before/after execution record.** This is enforced in code by `hooks/verification.py`, not by convention. A `Finding` object cannot be constructed without both run IDs. If a code path ever needs to bypass that hook, the code path is wrong. This rule *is* the product.

**2. Never let the agent write a value it invented.** Every patch is a removal of information the strategy was not entitled to — a shift, a lag, a masked row. If a proposed patch adds a number that was not already in the data, reject it. Suggesting improvements to the strategy is explicitly out of scope for the same reason: advice cannot be execution-proven, and shipping unprovable claims inside a tool whose thesis is "prove everything" would sink the whole submission.

**3. Distinguish three failure modes, always.** "Leak proven," "my patch was broken," and "no trades on this data" look similar from the outside and must never be collapsed. A strategy that produces zero trades after patching has not been proven clean — it has been proven untestable, and must be reported as such.

**4. Superpowers and ECC never appear in runtime code, imports, `requirements.txt`, or `REPRODUCE.md`.** See §5.1.

**5. Keep the data small — but not too small.** Three years of daily bars, cached to disk and committed. One year was the earlier plan and it is not enough: the standard error of an annualised Sharpe on one year of daily data is roughly 1.0, which swamps most of the effects being measured, and it leaves no room for the three-window robustness check in the taxonomy's validity rules. Three years costs almost nothing in compute and is still a small cache. Twenty strategies × 2 runs per hypothesis × several hypotheses adds up fast, and a slow eval loop will quietly kill iteration speed on Sunday.

**6. Freeze the 20 eval cases before writing any agent code.** Cases edited after seeing results are not evidence. The 8 clean controls are as important as the 12 injected cases — a detector that flags everything scores zero on the false-positive metric.

**7. Draw leak patterns from the published taxonomy, not from imagination.** Cases sourced from Kapoor & Narayanan's leakage taxonomy, plus Freqtrade issue #11346, which is a real bug the official tool passes as clean and which we did not design. Self-designed cases invite "you built the test to pass."

**8. Sandbox everything, with a timeout.** Audited code is untrusted and often broken. Subprocess with a hard timeout and resource caps. A hung backtest must not hang the agent loop.

**9. Do not build:** login, accounts, database, Supabase, custom data upload, GitHub repo scanning, CI integration, auto-reoptimisation, strategy editor. Each scores zero points and several actively cost points — auth in particular adds signup friction at the exact moment that should be frictionless, and adds env vars to the reproduction path.

**10. Cut from the bottom of the build order, never from the middle.** If time runs short: video gets tighter, docs get terser, the UI falls back to screens 1 and 3. The eval harness and the agent loop are never what gets cut — they carry 45 of the 100 points between them.

**11. Report the numbers that hurt.** The changelog must include the experiment that was removed (LLM self-critique of its own findings) and the "what remains unproven" section — zero slippage assumed, survivorship-biased universe, optimised on the same data it is tested on. These are factual gaps, not opinions, and stating them is worth more than hiding them.
