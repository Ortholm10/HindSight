export default function SampleCard({ name, description, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="group relative w-full rounded-t-md border border-panel-border bg-panel px-6 py-5 text-left transition-colors hover:border-amber focus-visible:border-amber"
    >
      <span
        className="absolute -top-3 left-6 rounded-t-sm border border-b-0 border-panel-border bg-panel px-3 py-1 font-mono text-[11px] uppercase tracking-widest text-ink-muted group-hover:text-amber"
        aria-hidden="true"
      >
        case file
      </span>
      <h3 className="text-lg font-semibold text-ink">{name}</h3>
      <p className="mt-1 text-sm text-ink-muted">{description}</p>
    </button>
  );
}
