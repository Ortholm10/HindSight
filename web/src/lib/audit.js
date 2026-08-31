// Real audit client: uploads a file to the FastAPI server, opens the SSE
// stream, and dispatches the same seven event types the server relays
// straight from hindsight_core — nothing here reinterprets them.

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

const EVENT_TYPES = [
  "scan_complete",
  "triage",
  "baseline",
  "prove_start",
  "prove_result",
  "agent_decision",
  "final",
];

async function fetchRun(runId) {
  const res = await fetch(`${API_BASE}/runs/${runId}`);
  if (!res.ok) throw new Error(`run ${runId} not found (${res.status})`);
  return res.json();
}

function displayName(filename) {
  return filename.replace(/\.(py|ipynb)$/i, "").replace(/[_-]+/g, " ");
}

/**
 * Starts a real audit: POST /audit, then stream GET /stream/{job_id}.
 *
 * onEvent({type, payload}) fires for every relayed core event, in the same
 * shape LiveAudit's reducer already expects. onFinal(scenario) fires once,
 * after the "final" event, with before/after equity resolved via /runs/{id}
 * — the run_ids come straight off the final payload, never invented.
 */
export function startAudit(file, { mode = "agent", timeoutS = 60, onEvent, onFinal, onError }) {
  let stopped = false;
  let finished = false;
  let es = null;
  const baselineRunIds = [];

  async function run() {
    const form = new FormData();
    form.append("file", file, file.name);

    let jobId;
    try {
      const res = await fetch(
        `${API_BASE}/audit?mode=${encodeURIComponent(mode)}&timeout_s=${timeoutS}`,
        { method: "POST", body: form }
      );
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`upload failed (${res.status}): ${detail}`);
      }
      ({ job_id: jobId } = await res.json());
    } catch (err) {
      if (!stopped) onError(err.message);
      return;
    }

    if (stopped) return;
    es = new EventSource(`${API_BASE}/stream/${jobId}`);

    for (const type of EVENT_TYPES) {
      es.addEventListener(type, (e) => {
        if (stopped) return;
        const payload = JSON.parse(e.data);
        onEvent({ type, payload });

        if (type === "baseline") baselineRunIds.push(payload.run_id);
        if (type === "final") {
          // The server closes the stream right after this event — the browser's
          // own reconnect-on-close then fires a spurious "error" for a stream
          // that already delivered everything. Mark it done before that lands.
          finished = true;
          handleFinal(payload).catch((err) => onError(err.message));
        }
      });
    }

    es.onerror = () => {
      if (stopped || finished) return;
      onError("stream disconnected before the audit finished");
      es.close();
    };
  }

  async function handleFinal(payload) {
    const findings = payload.findings ?? [];
    const beforeRunId = baselineRunIds[0];
    const afterRunId =
      findings.length > 0
        ? findings[findings.length - 1].after_run_id
        : baselineRunIds[baselineRunIds.length - 1];

    const [before, after] = await Promise.all([
      beforeRunId ? fetchRun(beforeRunId).catch(() => null) : null,
      afterRunId ? fetchRun(afterRunId).catch(() => null) : null,
    ]);

    if (stopped) return;
    onFinal({
      displayName: displayName(file.name),
      description: `${mode} audit of ${file.name}`,
      verdict: payload.verdict,
      reason: payload.reason,
      findings,
      unproven: payload.unproven ?? [],
      budget: payload.budget,
      reportedSharpe: payload.original_metrics?.sharpe ?? before?.metrics?.sharpe ?? null,
      actualSharpe: payload.baseline_metrics?.sharpe ?? after?.metrics?.sharpe ?? null,
      beforeEquity: before?.equity ?? [],
      afterEquity: after?.equity ?? [],
    });
    es.close();
  }

  run();

  return function stop() {
    stopped = true;
    if (es) es.close();
  };
}
