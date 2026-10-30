"""Operations-research view of the maintenance decision (no model training; uses saved run outputs).

1. Cost model (age-replacement / renewal-reward): each engine either gets a planned service at cycle s (cost 1)
   or fails first (cost r = failure cost / planned-service cost). Cost rate = total cost / total cycles operated,
   so both failures and early service (useful life thrown away) are penalised in one number.
2. Optimal fixed interval T*: the classic age-replacement policy, T chosen to minimise the cost rate on the
   fleet's own history (non-validation training engines), then evaluated on validation engines.
3. Cost-optimal agent thresholds: the agent's service trigger (HIGH: RUL <= high_rul or P(fail) >= 0.5;
   CRITICAL: lower bound <= critical; due = lower bound - buffer) tuned by 2-fold cross-validation over the
   validation engines, so tuned thresholds are always scored on engines they weren't tuned on.
4. Lead time: a service planned fewer than `lead` cycles before failure can't be carried out (parts, slots) and
   counts as a failure. Without it the optimiser simply cuts the safety margin to whatever these few engines allow.

Findings (run v3): the agent beats the cost-optimal fixed interval at every cost ratio; CV-tuned thresholds do not
generalise reliably on 20-52 engines per subset (out-of-fold failures at a 10-cycle lead time), so the current
thresholds are kept; the safety buffer should equal the real maintenance lead time.

    python scripts/cost_optimization.py [run_dir]     # default: output/all_run_v3
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output" / "all_run_v3"
DATA = ROOT / "data" / "csv"
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
RATIOS = [3, 5, 10, 20]
LEAD_TIMES = [0, 10, 15]
GRID = {"high_rul": [20, 25, 30, 35, 40, 50, 60], "critical_rul_lower": [5, 10, 15, 20, 25], "safety_buffer": [0, 5, 10, 15, 20]}


def cost_rate(service, lives, ratio, lead=0):
    # renewal-reward cost per 1000 operated cycles; a service later than life - lead can't be carried out in time
    cost, cycles, fails = 0.0, 0, 0
    for s, life in zip(service, lives):
        if s is None or s >= life or s > life - lead:
            cost, cycles, fails = cost + ratio, cycles + life, fails + 1
        else:
            cost, cycles = cost + 1.0, cycles + s
    return 1000 * cost / cycles, fails


def life_used(service, lives):
    return 100 * np.mean([1.0 if s is None or s >= l else s / l for s, l in zip(service, lives)])


def service_cycle(e, high_rul, critical, buffer, high_p=0.5):
    # cycle the agent's first HIGH/CRITICAL order falls due (mirrors TriageAgent: CRITICAL due now,
    # HIGH due at lower bound - buffer; a later CRITICAL makes the order due immediately)
    cyc, pred, lo, p = e["cycle"], e["rul_pred"], e["rul_lo"], e["p_fail"]
    crit = lo <= critical
    high = (pred <= high_rul) | (p >= high_p) | crit
    if not high.any():
        return None
    i = int(np.argmax(high))
    due = cyc[i] if crit[i] else cyc[i] + max(0, int(np.floor(lo[i])) - buffer)
    later_crit = np.flatnonzero(crit[i:])
    if len(later_crit):
        due = min(due, cyc[i + later_crit[0]])
    return int(due)


def engines(P):
    return {u: {c: g[c].to_numpy() for c in ["cycle", "rul_pred", "rul_lo", "p_fail"]}
            for u, g in P.sort_values(["unit", "cycle"]).groupby("unit")}


def best_thresholds(E, units, lives, ratio, lead):
    best = None
    for h, c, b in itertools.product(*GRID.values()):
        rate, _ = cost_rate([service_cycle(E[u], h, c, b) for u in units], [lives[u] for u in units], ratio, lead)
        if best is None or rate < best[0]:
            best = (rate, {"high_rul": h, "critical_rul_lower": c, "safety_buffer": b})
    return best[1]


def optimal_T(hist_lives, ratio, lead):
    grid = range(50, int(max(hist_lives)) + 1)
    return min(grid, key=lambda t: cost_rate([t] * len(hist_lives), hist_lives, ratio, lead)[0])


def main():
    rows, recommended = [], {}
    rng = np.random.default_rng(42)
    for subset in SUBSETS:
        mon = json.loads((RUN / "artifacts" / subset / "monitoring.json").read_text())
        cfg = mon["agent_config"]
        val = set(mon["splits"]["val"])
        life = pd.read_csv(DATA / f"train_{subset}.csv").groupby("unit").cycle.max()
        hist = [int(x) for x in life[~life.index.isin(val)]]
        P = pd.read_csv(RUN / "artifacts" / subset / "predictions_val.csv")
        E = engines(P)
        units = sorted(E)
        lives = {u: int(life[u]) for u in units}
        L = [lives[u] for u in units]
        default = [service_cycle(E[u], cfg["high_rul"], cfg["critical_rul_lower"], cfg["safety_buffer"]) for u in units]
        folds = np.array_split(rng.permutation(units), 2)
        for lead, r in itertools.product(LEAD_TIMES, RATIOS):
            t_star = optimal_T(hist, r, lead)
            tuned = {}
            for k in range(2):
                train_u, test_u = list(folds[1 - k]), list(folds[k])
                th = best_thresholds(E, train_u, lives, r, lead)
                for u in test_u:
                    tuned[u] = service_cycle(E[u], th["high_rul"], th["critical_rul_lower"], th["safety_buffer"])
            policies = {
                "reactive": [None] * len(units),
                "optimal fixed interval T*": [t_star] * len(units),
                "agent, current thresholds": default,
                "agent, cost-tuned thresholds (2-fold CV)": [tuned[u] for u in units],
            }
            for name, svc in policies.items():
                rate, fails = cost_rate(svc, L, r, lead)
                rows.append({"subset": subset, "lead": lead, "ratio": r, "policy": name, "t_star": t_star,
                             "cost_per_1000_cycles": rate, "unplanned_failures": fails, "engines": len(units),
                             "life_used_pct": life_used(svc, L)})
            recommended[(subset, lead, r)] = best_thresholds(E, units, lives, r, lead)
    table = pd.DataFrame(rows)
    return table, recommended


if __name__ == "__main__":
    table, recommended = main()
    pd.set_option("display.width", 200)
    view = table.pivot_table(index=["lead", "ratio", "subset"], columns="policy",
                             values=["cost_per_1000_cycles", "unplanned_failures"], aggfunc="first")
    print(view.round(2).to_string())
    out = RUN / "artifacts" / "cost_optimization.json"
    out.write_text(json.dumps({"table": table.to_dict(orient="records"),
                               "recommended_thresholds": {f"{s}_lead{l}_ratio{r}": v for (s, l, r), v in recommended.items()},
                               "grid": GRID, "ratios": RATIOS, "lead_times": LEAD_TIMES}, indent=2))
    print("\nwritten:", out)
