import { useState } from "react";
import DropZone from "./components/DropZone";
import LiveAudit from "./components/LiveAudit";
import Verdict from "./components/Verdict";

export default function App() {
  const [screen, setScreen] = useState("drop"); // drop | audit | verdict
  const [file, setFile] = useState(null);
  const [scenario, setScenario] = useState(null);
  const [streamError, setStreamError] = useState(null);

  function selectFile(f) {
    setFile(f);
    setStreamError(null);
    setScreen("audit");
  }

  function restart() {
    setScreen("drop");
    setFile(null);
    setScenario(null);
    setStreamError(null);
  }

  return (
    <div className="min-h-screen bg-bg text-ink">
      {screen === "drop" && <DropZone onSelect={selectFile} />}

      {screen === "audit" && file && !streamError && (
        <LiveAudit
          file={file}
          mode="agent"
          onDone={(result) => {
            setScenario(result);
            setScreen("verdict");
          }}
          onError={(msg) => setStreamError(msg)}
        />
      )}

      {screen === "audit" && streamError && (
        <div className="flex h-screen flex-col items-center justify-center gap-4 px-6 text-center">
          <p className="font-mono text-xs uppercase tracking-widest text-diff-remove">
            stream error
          </p>
          <p className="max-w-md text-ink-muted">{streamError}</p>
          <button
            type="button"
            onClick={restart}
            className="rounded-sm border border-panel-border px-4 py-2 font-mono text-xs text-ink hover:border-amber"
          >
            ← back to start
          </button>
        </div>
      )}

      {screen === "verdict" && scenario && <Verdict scenario={scenario} onRestart={restart} />}
    </div>
  );
}
