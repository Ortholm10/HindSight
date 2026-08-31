import { useEffect, useRef, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, Tooltip, XAxis, YAxis } from "recharts";

// Recharts' own ResponsiveContainer sizes itself off a ResizeObserver whose
// first callback can report a stale/zero box right after this chart mounts
// as part of a full screen swap (audit -> verdict) — Chromium sometimes
// doesn't re-fire until a later, unrelated resize (e.g. a scrollbar
// appearing). Measuring the wrapper div's clientWidth/clientHeight directly
// in an effect reads the already-committed layout synchronously, so the
// first paint is correct without waiting on that callback.
function useElementSize() {
  const ref = useRef(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    setSize({ width: el.clientWidth, height: el.clientHeight });
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ width, height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return [ref, size];
}

function mergeCurves(before, after) {
  const map = new Map();
  for (const [date, v] of before) map.set(date, { date, reported: v });
  for (const [date, v] of after) {
    const row = map.get(date) ?? { date };
    row.actual = v;
    map.set(date, row);
  }
  return Array.from(map.values());
}

function sameCurve(before, after) {
  if (before.length !== after.length) return false;
  return before.every(([, v], i) => Math.abs(v - after[i][1]) < 1e-9);
}

export default function EquityChart({ beforeEquity, afterEquity, identical }) {
  const [containerRef, { width, height }] = useElementSize();
  const data = mergeCurves(beforeEquity, identical ? [] : afterEquity);

  return (
    <div
      ref={containerRef}
      className="h-80 w-full rounded-md border border-panel-border bg-panel p-4"
    >
      {width > 0 && height > 0 && (
        <LineChart
          width={width}
          height={height}
          data={data}
          margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
        >
          <CartesianGrid stroke="var(--color-panel-border)" strokeDasharray="3 3" />
          <XAxis
            dataKey="date"
            tick={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--color-ink-muted)" }}
            minTickGap={60}
          />
          <YAxis
            tick={{ fontFamily: "var(--font-mono)", fontSize: 10, fill: "var(--color-ink-muted)" }}
            width={48}
          />
          <Tooltip
            contentStyle={{
              background: "var(--color-bg)",
              border: "1px solid var(--color-panel-border)",
              fontFamily: "var(--font-mono)",
              fontSize: 12,
            }}
            labelStyle={{ color: "var(--color-ink-muted)" }}
          />
          <Legend wrapperStyle={{ fontFamily: "var(--font-mono)", fontSize: 12 }} />
          <Line
            type="monotone"
            dataKey="reported"
            name="Reported"
            stroke="var(--color-slate)"
            dot={false}
            strokeWidth={2}
          />
          {!identical && (
            <Line
              type="monotone"
              dataKey="actual"
              name="Actual"
              stroke="var(--color-amber)"
              dot={false}
              strokeWidth={2}
            />
          )}
        </LineChart>
      )}
    </div>
  );
}

export { sameCurve };
