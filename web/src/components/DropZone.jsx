import { useRef, useState } from "react";
import SampleCard from "./SampleCard";

const SAMPLES = [
  {
    key: "golden_cross",
    name: "Golden Cross",
    description: "Classic dual moving-average crossover.",
    file: "/samples/golden_cross.py",
  },
  {
    key: "rsi_mean_reversion",
    name: "RSI Mean Reversion",
    description: "Oscillator threshold entries.",
    file: "/samples/rsi_mean_reversion.py",
  },
  {
    key: "multi_timeframe_momentum",
    name: "Multi-Timeframe Momentum",
    description: "Daily entries confirmed on a higher timeframe.",
    file: "/samples/multi_timeframe_momentum.py",
  },
];

async function fetchSample(sample) {
  const res = await fetch(sample.file);
  const blob = await res.blob();
  return new File([blob], `${sample.key}.py`, { type: "text/x-python" });
}

export default function DropZone({ onSelect }) {
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef(null);

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) onSelect(dropped);
  }

  function handleFileInput(e) {
    const picked = e.target.files?.[0];
    if (picked) onSelect(picked);
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
        onClick={() => fileInputRef.current?.click()}
        className={`flex cursor-pointer flex-col items-center gap-2 rounded-md border-2 border-dashed px-8 py-14 text-center transition-colors ${
          dragging ? "border-amber bg-amber-dim/30" : "border-panel-border"
        }`}
      >
        <p className="font-mono text-sm text-ink-muted">
          drop a .py file here, or click to browse
        </p>
        <input
          ref={fileInputRef}
          type="file"
          accept=".py"
          onChange={handleFileInput}
          className="hidden"
        />
      </div>

      <div className="grid gap-6 sm:grid-cols-3">
        {SAMPLES.map((s) => (
          <SampleCard
            key={s.key}
            name={s.name}
            description={s.description}
            onClick={() => fetchSample(s).then(onSelect)}
          />
        ))}
      </div>
    </div>
  );
}
