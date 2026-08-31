import StatusBadge from "./StatusBadge";
import EquityChart, { sameCurve } from "./EquityChart";
import FindingCard from "./FindingCard";

const UNPROVEN_GAPS = [
  "Zero slippage is assumed on every simulated fill.",
  "The universe is survivorship-biased — delisted symbols are not in the data.",
  "The strategy was optimised on the same data it was tested on.",
];

function fmtSharpe(v) {
  return v === null || v === undefined ? "—" : v.toFixed(2);
}

function downloadText(filename, mime, text) {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function buildPatchedPy(findings) {
  const header = "# Mock data — hindsight_server will serve the real patched file\n\n";
  return header + findings.map((f) => f.diff).join("\n\n");
}

function buildReportMd(scenario) {
  const lines = [
    `# Hindsight audit report`,
    ``,
    `**${scenario.displayName}** — verdict: \`${scenario.verdict}\``,
    ``,
    scenario.reason,
    ``,
    `## Findings (${scenario.findings.length})`,
  ];
  for (const f of scenario.findings) {
    lines.push(
      `- ${f.candidate.leak_type} at ${f.candidate.file}:${f.candidate.line} — before ${f.before_run_id}, after ${f.after_run_id}`
    );
  }
  lines.push("", `## Examined and unproven (${scenario.unproven.length})`);
  for (const u of scenario.unproven) {
    lines.push(`- ${u.candidate.leak_type}: ${u.status} — ${u.reason}`);
  }
  lines.push("", `## What this audit does not prove`, ...UNPROVEN_GAPS.map((g) => `- ${g}`));
  return lines.join("\n");
}

export default function Verdict({ scenario, onRestart }) {
  const identical = sameCurve(scenario.beforeEquity, scenario.afterEquity);

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-8 px-6 py-12">
      <div className="flex items-center justify-between">
        <p className="font-mono text-xs uppercase tracking-widest text-ink-muted">
          {scenario.displayName}
        </p>
        <button
          type="button"
          onClick={onRestart}
          className="font-mono text-xs text-ink-muted underline decoration-dotted hover:text-amber"
        >
          ← new audit
        </button>
      </div>

      <StatusBadge verdict={scenario.verdict} />

      <p className="text-ink-muted">{scenario.reason}</p>

      <div className="font-mono text-5xl font-bold text-ink sm:text-6xl">
        {fmtSharpe(scenario.reportedSharpe)}
        <span className="mx-4 text-ink-muted">→</span>
        <span className={scenario.verdict === "clean" ? "text-green" : "text-amber"}>
          {fmtSharpe(scenario.actualSharpe)}
        </span>
      </div>

      <EquityChart
        beforeEquity={scenario.beforeEquity}
        afterEquity={scenario.afterEquity}
        identical={identical}
      />
      {identical && (
        <p className="-mt-4 font-mono text-xs text-ink-muted">
          no repair — reported and actual are the same run
        </p>
      )}

      {scenario.findings.length > 0 && (
        <section>
          <h2 className="mb-3 font-mono text-xs uppercase tracking-widest text-ink-muted">
            Findings ({scenario.findings.length})
          </h2>
          <div className="flex flex-col gap-3">
            {scenario.findings.map((f, i) => (
              <FindingCard key={i} finding={f} />
            ))}
          </div>
        </section>
      )}

      {scenario.unproven.length > 0 && (
        <section>
          <h2 className="mb-3 font-mono text-xs uppercase tracking-widest text-ink-muted">
            Examined and unproven ({scenario.unproven.length})
          </h2>
          <div className="flex flex-col gap-2">
            {scenario.unproven.map((u, i) => (
              <div key={i} className="rounded-sm border border-panel-border bg-panel p-3">
                <div className="font-mono text-xs text-slate">
                  {u.candidate.leak_type} · {u.status}
                </div>
                <p className="mt-1 text-sm text-ink-muted">{u.reason}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="border-t border-panel-border pt-6">
        <h2 className="mb-3 font-mono text-xs uppercase tracking-widest text-ink-muted">
          What this audit does not prove
        </h2>
        <ul className="flex flex-col gap-2 text-sm text-ink-muted">
          {UNPROVEN_GAPS.map((g) => (
            <li key={g}>{g}</li>
          ))}
        </ul>
      </section>

      <div className="flex flex-wrap gap-3 border-t border-panel-border pt-6">
        <button
          type="button"
          onClick={() => downloadText("patched_strategy.py", "text/x-python", buildPatchedPy(scenario.findings))}
          disabled={scenario.findings.length === 0}
          className="rounded-sm border border-panel-border px-4 py-2 font-mono text-xs text-ink hover:border-amber disabled:cursor-not-allowed disabled:opacity-40"
        >
          Download patched .py
        </button>
        <button
          type="button"
          onClick={() => downloadText("audit_report.md", "text/markdown", buildReportMd(scenario))}
          className="rounded-sm border border-panel-border px-4 py-2 font-mono text-xs text-ink hover:border-amber"
        >
          Download audit report .md
        </button>
      </div>
    </div>
  );
}
