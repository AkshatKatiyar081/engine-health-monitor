"""Compare maintenance policies on validation engines: reactive, fixed interval, and our predictive agent.

Uses the saved outputs of run_all_subsets.ipynb (no model training) and the exact TriageAgent from the
generated pipeline notebook. Validation engines run to failure and were never trained on by the champion.

    python scripts/policy_comparison.py [run_dir]      # default: output/all_run_v2

Policies
- reactive: run to failure; every failure is unplanned, no useful life wasted.
- fixed interval T: service every engine at cycle T. Engines with life <= T fail unplanned; others waste life - T.
  T is set on the non-validation training engines (as a fleet would, from its own history): the largest T with
  no failures in that history, and the T allowing 5% failures.
- predictive (ours): replay the agent cycle by cycle on the champion's predictions; service happens at the due
  cycle of the first HIGH or CRITICAL work order (at_cycle + due_within_cycles). If that is at or after the
  engine's last cycle, the engine failed before service.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output" / "all_run_v2"
DATA = ROOT / "data" / "csv"


def load_agent():
    nb = json.loads((ROOT / "notebooks" / "pipeline_FD001.ipynb").read_text(encoding="utf-8"))
    src = next("".join(c["source"]) for c in nb["cells"]
               if c["cell_type"] == "code" and "class TriageAgent" in "".join(c["source"]))
    ns = {"np": np, "pd": pd, "CFG": {"fail_window": 30},
          "now_iso": lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")}
    exec(src, ns)
    return ns["TriageAgent"], ns["AGENT_CFG"]


def predictive_service(agent_cls, cfg, subset, engine):
    # cycle at which the agent's first HIGH/CRITICAL order falls due; None if it never raises one
    e = engine.sort_values("cycle").copy()
    e["anomaly_score"], e["recent_imputed"] = 0.0, False
    e["cycles_observed"] = np.arange(1, len(e) + 1)
    agent = agent_cls(cfg, subset, "replay")
    due = []
    for _, r in e.iterrows():
        ev = agent.process(r)
        if ev["event"] in ("created", "escalated") and ev["severity"] in ("HIGH", "CRITICAL"):
            wo = agent.work_orders[-1]
            due.append(wo["at_cycle"] + wo["due_within_cycles"])
    return min(due) if due else None


def evaluate(service_cycles, lives):
    # service_cycles: planned service cycle per engine (None = never); lives: failure cycle per engine
    failed = [s is None or s >= life for s, life in zip(service_cycles, lives)]
    wasted = [0 if f else life - s for s, life, f in zip(service_cycles, lives, failed)]
    used = [1.0 if f else s / life for s, life, f in zip(service_cycles, lives, failed)]
    return {"unplanned_failures": int(sum(failed)), "engines": len(lives),
            "mean_cycles_wasted": float(np.mean(wasted)), "life_used_pct": float(np.mean(used) * 100)}


def main():
    agent_cls, base_cfg = load_agent()
    rows, curves = [], {}
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        mon = json.loads((RUN / "artifacts" / subset / "monitoring.json").read_text())
        cfg = {**base_cfg, **mon["agent_config"]}
        val_units = set(mon["splits"]["val"])
        train = pd.read_csv(DATA / f"train_{subset}.csv")
        life = train.groupby("unit").cycle.max()
        hist_life, val_life = life[~life.index.isin(val_units)], life[life.index.isin(val_units)]

        preds = pd.read_csv(RUN / "artifacts" / subset / "predictions_val.csv")
        units = sorted(val_life.index)
        lives = [int(val_life[u]) for u in units]

        ours = evaluate([predictive_service(agent_cls, cfg, subset, preds[preds.unit == u]) for u in units], lives)
        reactive = evaluate([None] * len(units), lives)
        t_safe = int(hist_life.min()) - 1           # no failure in the fleet's own history
        t_5pct = int(np.quantile(hist_life, 0.05))   # 5% of historical engines would fail first
        fixed_safe = evaluate([t_safe] * len(units), lives)
        fixed_5 = evaluate([t_5pct] * len(units), lives)
        curves[subset] = {int(t): evaluate([int(t)] * len(units), lives) for t in range(100, int(max(lives)) + 1, 10)}

        for name, res in [("reactive (run to failure)", reactive), (f"fixed interval T={t_safe} (0% history failures)", fixed_safe),
                          (f"fixed interval T={t_5pct} (5% history failures)", fixed_5), ("predictive agent (ours)", ours)]:
            rows.append({"subset": subset, "policy": name, **res})

    table = pd.DataFrame(rows)
    table["mean_cycles_wasted"] = table.mean_cycles_wasted.round(1)
    table["life_used_pct"] = table.life_used_pct.round(1)
    print(table.to_string(index=False))
    out = RUN / "artifacts" / "policy_comparison.json"
    out.write_text(json.dumps({"table": table.to_dict(orient="records"), "fixed_interval_curves": curves}, indent=2))
    print("\nwritten:", out)


if __name__ == "__main__":
    main()
