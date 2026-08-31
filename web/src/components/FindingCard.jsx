import { useState } from "react";

function DiffLine({ line }) {
  let color = "text-ink-muted";
  if (line.startsWith("+") && !line.startsWith("+++")) color = "text-diff-add";
  if (line.startsWith("-") && !line.startsWith("---")) color = "text-diff-remove";
  return <div className={color}>{line || " "}</div>;
}

export default function FindingCard({ finding }) {
  const [open, setOpen] = useState(false);
  const { candidate, before_run_id, after_run_id, diff, delta } = finding;

  return (
    <div className="rounded-md border border-amber/40 bg-panel">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-4 px-4 py-3 text-left"
      >
        <div>
          <div className="font-mono text-xs text-amber">
            {candidate.leak_type} · {candidate.file.split(/[\\/]/).pop()}:{candidate.line}
          </div>
          <p className="mt-1 text-sm text-ink">{candidate.reason}</p>
        </div>
        <span className="font-mono text-xs text-ink-muted">{open ? "hide diff" : "show diff"}</span>
      </button>

      <div className="flex flex-wrap gap-4 border-t border-panel-border px-4 py-2 font-mono text-[11px] text-ink-muted">
        <span>before {before_run_id}</span>
        <span>after {after_run_id}</span>
        {delta?.sharpe !== undefined && (
          <span className="text-amber">Δsharpe {delta.sharpe.toFixed(3)}</span>
        )}
      </div>

      {open && (
        <pre className="overflow-x-auto border-t border-panel-border px-4 py-3 font-mono text-xs leading-relaxed">
          {diff.split("\n").map((l, i) => (
            <DiffLine key={i} line={l} />
          ))}
        </pre>
      )}
    </div>
  );
}
