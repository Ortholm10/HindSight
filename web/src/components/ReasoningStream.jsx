import { useEffect, useRef, useState } from "react";

const ACTION_STYLE = {
  accept: "text-amber",
  discard: "text-ink-muted",
  unproven: "text-slate",
};

export default function ReasoningStream({ decisions }) {
  const scrollRef = useRef(null);
  const [following, setFollowing] = useState(true);

  useEffect(() => {
    if (!following || !scrollRef.current) return;
    scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [decisions, following]);

  function handleScroll(e) {
    const el = e.currentTarget;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    setFollowing(distanceFromBottom < 80);
  }

  return (
    <div
      ref={scrollRef}
      onScroll={handleScroll}
      className="h-full overflow-y-auto px-5 py-4"
    >
      <h2 className="mb-3 font-mono text-xs uppercase tracking-widest text-ink-muted">
        Agent reasoning
      </h2>
      <div className="flex flex-col gap-3">
        {decisions.length === 0 && (
          <p className="font-mono text-sm text-ink-muted">waiting for the first decision…</p>
        )}
        {decisions.map((d, i) => (
          <div key={i} className="border-l-2 border-panel-border pl-3">
            <div className="flex flex-wrap items-baseline gap-2 font-mono text-xs text-ink-muted">
              <span>step {d.step}</span>
              <span className={`font-semibold uppercase ${ACTION_STYLE[d.action] ?? "text-ink"}`}>
                {d.action}
              </span>
              {d.plausible === false && (
                <span className="text-amber">still implausible — continuing</span>
              )}
            </div>
            <p className="mt-1 text-sm leading-relaxed text-ink">{d.reason}</p>
            <div className="mt-1 flex flex-wrap gap-3 font-mono text-[11px] text-ink-muted">
              <span>sharpe {d.sharpe?.toFixed(3) ?? "—"}</span>
              <span>findings {d.findings}</span>
              <span>pending {d.pending}</span>
              <span>llm {d.llm_calls}</span>
              <span>sandbox {d.sandbox_runs}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
