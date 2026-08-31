const VARIANTS = {
  leaks_proven: {
    label: "LEAK PROVEN",
    color: "text-amber border-amber",
    glow: "shadow-[0_0_24px_-6px_var(--color-amber)]",
  },
  clean: {
    label: "NO LEAK PROVEN",
    color: "text-green border-green",
    glow: "shadow-[0_0_24px_-6px_var(--color-green)]",
  },
  untestable: {
    label: "UNTESTABLE",
    color: "text-slate border-slate",
    glow: "",
  },
  inconclusive: {
    label: "INCONCLUSIVE",
    color: "text-slate border-slate",
    glow: "",
  },
  stopped_on_budget: {
    label: "INCOMPLETE — STOPPED ON BUDGET",
    color: "text-amber border-amber border-dashed",
    glow: "",
  },
};

export default function StatusBadge({ verdict }) {
  const v = VARIANTS[verdict] ?? VARIANTS.inconclusive;
  return (
    <div
      className={`animate-stamp-in inline-flex items-center gap-3 rounded-sm border-4 px-6 py-3 font-mono text-xl font-bold tracking-widest ${v.color} ${v.glow}`}
      style={{ transform: "rotate(-4deg)" }}
    >
      {v.label}
    </div>
  );
}
