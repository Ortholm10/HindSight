import { useState } from "react";
import SampleCard from "./SampleCard";

const SAMPLES = [
  {
    key: "golden_cross",
    name: "Golden Cross",
    description: "Classic dual moving-average crossover.",
  },
  {
    key: "rsi_mean_reversion",
    name: "RSI Mean Reversion",
    description: "Oscillator threshold entries.",
  },
  {
    key: "multi_timeframe_momentum",
    name: "Multi-Timeframe Momentum",
    description: "Daily entries confirmed on a higher timeframe.",
  },
];

const EDGE_CASES = [
  { key: "edge_untestable", label: "untestable" },
  { key: "edge_budget", label: "stopped on budget" },
  { key: "edge_error", label: "dropped stream" },
];

export default function DropZone({ onSelect }) {
  const [dragging, setDragging] = useState(false);
  const [droppedNote, setDroppedNote] = useState(false);

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files?.length) {
      setDroppedNote(true);
      onSelect("golden_cross");
    }
  }

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-10 px-6 py-16">
      <header className="text-center">
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-amber">
          Hindsight
        </p>
        <h1 className="mt-3 text-3xl font-semibold text-ink">
          Audit a backtest for look-ahead bias
        </h1>
        <p className="mt-2 text-ink-muted">
          Drop a strategy file, or open a recorded case below.
        </p>
      </header>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={`flex flex-col items-center gap-2 rounded-md border-2 border-dashed px-8 py-14 text-center transition-colors ${
          dragging ? "border-amber bg-amber-dim/30" : "border-panel-border"
        }`}
      >
        <p className="font-mono text-sm text-ink-muted">
          drop a .py file or notebook here
        </p>
        {droppedNote && (
          <p className="mt-2 font-mono text-xs text-amber">
            No server yet — showing a recorded audit for this file.
          </p>
        )}
      </div>

      <div className="grid gap-6 sm:grid-cols-3">
        {SAMPLES.map((s) => (
          <SampleCard
            key={s.key}
            name={s.name}
            description={s.description}
            onClick={() => onSelect(s.key)}
          />
        ))}
      </div>

      <div className="text-center font-mono text-xs text-ink-muted">
        Edge cases:{" "}
        {EDGE_CASES.map((e, i) => (
          <span key={e.key}>
            {i > 0 && " · "}
            <button
              type="button"
              onClick={() => onSelect(e.key)}
              className="underline decoration-dotted underline-offset-4 hover:text-amber"
            >
              {e.label}
            </button>
          </span>
        ))}
      </div>
    </div>
  );
}
