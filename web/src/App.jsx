import { useState } from "react";
import DropZone from "./components/DropZone";
import LiveAudit from "./components/LiveAudit";
import Verdict from "./components/Verdict";

import goldenCross from "./mocks/golden_cross.json";
import rsiMeanReversion from "./mocks/rsi_mean_reversion.json";
import multiTimeframeMomentum from "./mocks/multi_timeframe_momentum.json";
import edgeUntestable from "./mocks/edge_untestable.json";
import edgeBudget from "./mocks/edge_budget.json";
import edgeError from "./mocks/edge_error.json";

const SCENARIOS = {
  golden_cross: goldenCross,
  rsi_mean_reversion: rsiMeanReversion,
  multi_timeframe_momentum: multiTimeframeMomentum,
  edge_untestable: edgeUntestable,
  edge_budget: edgeBudget,
  edge_error: edgeError,
};

export default function App() {
  const [screen, setScreen] = useState("drop"); // drop | audit | verdict
  const [scenarioKey, setScenarioKey] = useState(null);
  const [streamError, setStreamError] = useState(null);

  const scenario = scenarioKey ? SCENARIOS[scenarioKey] : null;

  function selectScenario(key) {
    setScenarioKey(key);
    setStreamError(null);
    setScreen("audit");
  }

  function restart() {
    setScreen("drop");
    setScenarioKey(null);
    setStreamError(null);
  }

  return (
    <div className="min-h-screen bg-bg text-ink">
      {screen === "drop" && <DropZone onSelect={selectScenario} />}

      {screen === "audit" && scenario && !streamError && (
        <LiveAudit
          scenario={scenario}
          onDone={() => setScreen("verdict")}
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
