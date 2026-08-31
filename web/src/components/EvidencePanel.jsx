function Section({ title, count, children }) {
  return (
    <section className="mb-6">
      <h3 className="mb-2 font-mono text-xs uppercase tracking-widest text-ink-muted">
        {title} <span className="text-ink">({count})</span>
      </h3>
      <div className="flex flex-col gap-2">{children}</div>
    </section>
  );
}

export default function EvidencePanel({ candidates, discarded, runs }) {
  return (
    <div className="h-full overflow-y-auto px-5 py-4">
      <Section title="Candidates" count={candidates.length}>
        {candidates.map((c, i) => (
          <div key={i} className="rounded-sm border border-panel-border bg-panel p-3">
            <div className="flex items-center justify-between font-mono text-[11px] text-ink-muted">
              <span className="text-amber">{c.leak_type}</span>
              <span>
                {c.file.split(/[\\/]/).pop()}:{c.line}
              </span>
            </div>
            <pre className="mt-1 overflow-x-auto font-mono text-xs text-ink">{c.snippet}</pre>
            <p className="mt-1 text-xs text-ink-muted">{c.reason}</p>
          </div>
        ))}
        {candidates.length === 0 && (
          <p className="font-mono text-xs text-ink-muted">none yet</p>
        )}
      </Section>

      <Section title="Discarded" count={discarded.length}>
        {discarded.map((d, i) => (
          <div key={i} className="rounded-sm border border-panel-border p-3">
            <div className="font-mono text-[11px] text-ink-muted">
              {d.candidate.leak_type} · line {d.candidate.line}
            </div>
            <p className="mt-1 text-xs text-ink-muted">{d.reason}</p>
          </div>
        ))}
        {discarded.length === 0 && (
          <p className="font-mono text-xs text-ink-muted">none yet</p>
        )}
      </Section>

      <Section title="Runs" count={runs.length}>
        {runs.map((r, i) => (
          <div
            key={i}
            className={`flex items-center justify-between rounded-sm border p-2 font-mono text-[11px] ${
              r.status === "proven" ? "border-amber text-amber" : "border-panel-border text-ink-muted"
            }`}
          >
            <span>{r.operation}</span>
            <span>{r.status}</span>
            <span>
              {r.before_run_id.slice(0, 8)}→{r.after_run_id.slice(0, 8)}
            </span>
            <span>
              Δsharpe {r.delta?.sharpe !== undefined ? r.delta.sharpe.toFixed(3) : "—"}
            </span>
          </div>
        ))}
        {runs.length === 0 && <p className="font-mono text-xs text-ink-muted">none yet</p>}
      </Section>
    </div>
  );
}
