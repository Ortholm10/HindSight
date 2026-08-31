import { useEffect, useReducer } from "react";
import { startAudit } from "../lib/audit";
import ReasoningStream from "./ReasoningStream";
import EvidencePanel from "./EvidencePanel";

const initialState = {
  candidates: [],
  discarded: [],
  decisions: [],
  runs: [],
  streamError: null,
  done: false,
};

function reducer(state, event) {
  switch (event.type) {
    case "scan_complete":
      return { ...state, candidates: event.payload.candidates };
    case "prove_result":
      return { ...state, runs: [...state.runs, event.payload] };
    case "agent_decision": {
      const next = { ...state, decisions: [...state.decisions, event.payload] };
      if (event.payload.action === "discard") {
        next.discarded = [
          ...state.discarded,
          { candidate: event.payload.candidate, reason: event.payload.reason },
        ];
      }
      return next;
    }
    default:
      return state;
  }
}

export default function LiveAudit({ file, mode, onDone, onError }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  useEffect(() => {
    const stop = startAudit(file, {
      mode,
      onEvent: dispatch,
      onFinal: onDone,
      onError,
    });
    return stop;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file]);

  return (
    <div className="grid h-screen grid-cols-2 divide-x divide-panel-border">
      <ReasoningStream decisions={state.decisions} />
      <EvidencePanel
        candidates={state.candidates}
        discarded={state.discarded}
        runs={state.runs}
      />
    </div>
  );
}
