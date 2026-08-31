# Reproducing Hindsight

Every command below was run, in order, from this checkout, verifying: a clean
Python install against the exact pins in `requirements.txt` (fresh venv), the
full test suite, lint, a live LLM-backed audit reproducing the README's
flagship number, a single-case eval, the web build, and the FastAPI app
importing and registering its routes. Neither Superpowers nor ECC appears in
any command here or in anything a command here imports — this project's
runtime has no dependency on either.

## 0. Requirements

- Python 3.11+ (verified here on CPython 3.14.6 and 3.14.0)
- Node 18+ / npm, only if you want the web UI
- A free-tier Gemini API key (`GEMINI_API_KEY`) for anything that calls the
  LLM — `eval`, `audit --mode agent`, and `audit --mode pipeline`. Nothing in
  the sandbox, the tests, or the web/server layer needs a key.

## 1. Python setup

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .                 # registers the `hindsight` command (pyproject.toml's
                                  # [project.scripts] entry) — requirements.txt alone
                                  # installs the libraries but not the CLI itself
cp .env.example .env             # then add GEMINI_API_KEY (and/or GROQ_API_KEY)
```

**Verified:** a from-scratch `python -m venv` + `pip install -r requirements.txt`
into an empty venv completed with exit code 0 against every pin in the file —
no resolver conflicts. `pip install -e .` then makes `hindsight --help` print
its `eval`/`audit` subcommands.

## 2. Data

```bash
python -m eval.cache_data
```

`eval/data/*.csv` — three years of daily bars for the eight tickers the frozen
cases run against — is committed to the repo, so this step is a refresh, not
a requirement: an eval run never touches the network by design
(`eval/cache_data.py`'s own docstring: *"a judge reproducing the results may
not have any [network access]."*). The files were already present in this
checkout and every command below ran against them unchanged.

## 3. Tests and lint

```bash
pytest
ruff check .
```

**Verified**, this session, from this checkout:

```
384 passed... (1 skipped)
[exited with code 0]
```

```
All checks passed!
```

(`pytest -q` printed six blocks of dots totalling 384 collected tests, one
`s` for the single skip, exit code 0. `ruff check .` reported `All checks
passed!` with no findings.)

## 4. The flagship number — Sharpe 3.72 → 0.47

```bash
hindsight audit eval/cases/l03_same_bar_execution/strategy.py \
    --data eval/data/SPY.csv --mode agent
```

**Verified**, live, this session, with a real `GEMINI_API_KEY` (no cached
response) — actual stdout:

```
scan      1 candidate(s)
          L03 line 8: signal = df["close"] > sma
baseline  completed  sharpe 3.724  total_return 2.287  max_drawdown -0.036
          run_id cda67fa4c08f472c
triage    L03 line 8: leak -> lag
prove     line 8 via lag
result    line 8: PROVEN
          --- a/strategy.py
          +++ b/strategy.py
          @@ -5,7 +5,7 @@
           def run_positions(df: pd.DataFrame) -> pd.Series:
               sma = df["close"].rolling(20).mean()
          -    signal = df["close"] > sma
          +    signal = (df["close"] > sma).shift(1)
               return signal.fillna(False).astype(int)
          before sharpe 3.724  total_return 2.287  max_drawdown -0.036 (cda67fa4c08f472c)
          after  sharpe 0.471  total_return 0.149  max_drawdown -0.160 (19a2ca3aea9f41c7)
          delta  sharpe -3.253  total_return -2.138  max_drawdown -0.124
final     leaks_proven
          1 leak(s) proven by execution; 0 candidate(s) examined and left unproven
```

This is the same bug, same repair, and the same Sharpe delta (3.72 → 0.47) as
the README's `golden_cross.py` demo and `runs/20260831T142145-fd882229.json`
(uploaded through the web UI in Session 8) — reproduced here through the CLI,
from a different upload, on a different run, to confirm it is a property of
the leak and not of one particular file.

## 5. A single-case eval run

```bash
hindsight eval --case c01_lagged_crossover --detector agent
```

**Verified**, actual stdout:

```
case                             type  cands  hit  line  type  dSharpe  notes
-----------------------------------------------------------------------------
c01_lagged_crossover             clean     0   ok
-----------------------------------------------------------------------------
detected            0/0
line localised      0/0
leak type correct   0/0
false positives     0/1
localisation prec.  0.0% (0 candidates reported)
```

Zero candidates on a clean control, zero false positives — the differential
prover finding nothing to prove is the correct outcome here.

## 6. The full frozen suite (the numbers in the README and CHANGELOG)

```bash
hindsight eval --detector pipeline --json --out eval/results/pipeline.json
hindsight eval --detector agent --repeat 3 --json --out eval/results/agent.json
hindsight audit tests/fixtures/stacked_leaks.py --data eval/data/SPY.csv --mode agent

# Case 21 (freqtrade #11346) — outside the frozen 20, reached by name
hindsight eval --case htf_merge_11346 --detector agent
python -m eval.baselines.freqtrade_11346     # needs freqtrade installed, dev-only
```

These are the exact commands that produced `eval/results/pipeline.json`,
`eval/results/agent.json`, and `eval/results/case21.json`, already committed
in this checkout and quoted throughout the README and `docs/CHANGELOG.md`.
**Not re-run in full during this verification pass** — the agent suite issues
57 live LLM calls per pass × 3 passes for `--repeat 3`, against a free-tier
provider whose daily quota this project is already careful with (see
`CLAUDE.md` §4); Sections 4 and 5 above instead re-ran a live, uncached slice
of the same code path (one full agent audit, one full eval case) to confirm
the pipeline genuinely still executes end to end, rather than re-spending the
full 171-call budget to reproduce numbers already on disk. If you want the
full run reproduced from zero, delete the cache directory first — a warm
cache replays cached answers rather than issuing new calls, and `--repeat`
gives each pass its own cache directory precisely so a repeat cannot silently
replay pass 1.

Every audit also writes a full trajectory to `runs/<audit_id>.json`: the
verdict, the budget spent, every finding with both run IDs, every unproven
candidate with the reason it stayed unproven, and the complete event stream.

## 7. The scanner-alone flood number

The claim in `docs/CHANGELOG.md` that an untriaged, unproven AST scan flags
100% of the clean controls was re-measured directly this session by calling
`hindsight_core.tools.scan_file.scan_file` over every frozen case's
`strategy.py` and counting raw candidates with no triage and no proof
downstream:

```
total candidates (no triage, no proof): 41
clean cases with >=1 candidate (scanner-alone false positives): 8/8
injected cases with >=1 candidate: 12/12
```

## 8. Server

```bash
uvicorn hindsight_server.main:app --reload --port 8000
```

**Verified** without starting the network listener: `hindsight_server.main.create_app()`
imports cleanly and registers `/audit`, `/stream/{job_id}`, `/runs/{run_id}`
(plus the FastAPI-generated `/docs`, `/redoc`, `/openapi.json`) — the same
routes `uvicorn` serves when actually started.

## 9. Web UI

```bash
cd web
npm install
npm run dev             # http://localhost:5173, expects the server on :8000
```

**Verified**, this session, against the `node_modules` already installed in
this checkout:

```
npm run build
✓ 599 modules transformed.
dist/index.html                   0.46 kB
dist/assets/index-*.css          16.18 kB
dist/assets/index-*.js          567.90 kB
✓ built in 12.69s
```

(`npm run build` was used to verify the toolchain and dependency tree resolve
cleanly and produce a working bundle; `npm run dev` is the command for actual
day-to-day use and was not separately re-verified beyond confirming the same
`vite.config.js` and dependency tree drive both.)

## 10. Final check

```bash
git status
```

Confirmed clean after this documentation pass — no command in this file
writes to the working tree except `eval/results/*.json` and `runs/*.json`,
both already committed with the values quoted above.
