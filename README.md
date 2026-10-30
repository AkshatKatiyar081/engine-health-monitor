# Predictive maintenance for a connected fleet: from sensor stream to automatic work order

Predictive maintenance for a connected fleet on the NASA C-MAPSS turbofan data (all four subsets, FD001–FD004).

The system reads each machine's sensor history and predicts its **remaining useful life** (RUL, in operating cycles) with a calibrated 80% range.
It also estimates the probability of failure within 30 cycles and flags anomalies against the machine's "as new" behaviour.
A **rule-based triage agent** (plain Python, no LLM) turns these into **work orders automatically** and sends only exceptions to a human.
The MLOps loop tracks every run, versions the models, retrains on new data or drift, and promotes a new model only when it is reliably not worse.

**[Open the monitoring dashboard](index.html)** (one offline page; with GitHub Pages on it opens from the repo link).
Its *Live replay* tab streams held-out engines cycle by cycle and shows the agent raising, escalating and closing orders.

---

## Where every reported number comes from

All figures are from real runs of `notebooks/run_all_subsets.ipynb` on Kaggle (GPU T4).
The saved outputs are in `results/`; nothing was typed in by hand.

| Claim | What it means exactly | Evidence in this repo |
|---|---|---|
| Built and tested on all four subsets | One code path, run on FD001–FD004 without manual steps | `results/final_run/summary_all_subsets.csv`, `results/final_run/FD00x/monitoring.json` |
| Test RMSE 10.6–11.0 cycles | Official NASA test engines, RUL capped at 125 (standard for C-MAPSS). True-label RMSE: 11.7 / 23.1 / 12.5 / 23.7 | `summary_all_subsets.csv` (`test_rmse_capped`, `test_rmse`), `FD00x/predictions_test.csv` |
| Calibrated 80% range | Conformalised quantile models; actual coverage on test engines 84–85% | `monitoring.json` → `production_metrics.test_interval_coverage`, `cv_interval` |
| 142 held-out engines | Validation engines (20 + 52 + 20 + 50) never used for training, replayed cycle by cycle to their recorded failure | `monitoring.json` → `splits.val`, `FD00x/replay_log.csv` |
| 0 unplanned failures | The agent's first HIGH/CRITICAL order falls due before the engine's failure cycle, for every engine | `FD00x/fleet_engine_stats.csv` (`failed`, `cycles_left_at_service`), `monitoring.json` → `policy_comparison` |
| ~90% vs ~62% of useful life used | Service cycle ÷ failure cycle, averaged over engines. Fixed schedule = the longest interval with no failures in past training history (also 0 failures on these engines) | `monitoring.json` → `policy_comparison`, `policy_fixed_intervals` |
| Same result across 5 random seeds | Whole pipeline re-run with seeds 42, 1, 2, 3, 4: 0 failures and 0 missed near-failure engines in all 20 runs; life used 89–90% vs 62–65% | `results/seeds/summary_per_seed.csv`, `summary_robustness.csv` |
| Reproducible | A second Kaggle session on the same kind of machine gave identical results in all 12 runs, to every decimal. On a different Kaggle machine, GPU XGBoost shifted in the last digits: decisions identical, capped RMSE within ±0.3 (10.2–11.2) | `docs/methodology.md` §9.7, `results/gate_check/` |
| Promotion gate works | New non-inferiority gate promoted the retrained model in 12 of 12 runs, fixing the 3 runs the old gate wrongly blocked | `results/gate_check/summary_per_seed.csv`, `docs/methodology.md` §7.2 |
| Why not deep learning? | 1D-CNN, GRU and TCN (3 random starts each, same folds) were 3–6 cycles worse in CV on every subset; boosting kept | `results/deep/`, `docs/methodology.md` §9.9 |
| 22–47% lower maintenance cost | Renewal-reward cost rate vs the *cost-optimal* fixed interval, failure = 3–20× a planned service, maintenance lead time 0–10 cycles | `results/final_run/cost_optimization.json`, `scripts/cost_optimization.py`, `docs/methodology.md` §9.4 |
| Near-failure engines caught | 157 of 159 test engines with true RUL ≤ 30 got HIGH or CRITICAL; all 159 got an order; 5 false alarms (HIGH+ while true RUL > 100) | `summary_all_subsets.csv`, `monitoring.json` → `agent_summary`, `agent_confusion` |
| Automatic work orders | Every order the agent created, with severity, due date, reasons and review flags | `FD00x/work_orders.csv` / `.json` (e.g. `WO-FD004-008-0248`) |
| Sensor-health check | Stuck / dead / out-of-range sensors and unseen operating conditions are flagged; 16/16 injected faults caught; 0 false flags on real data | `FD00x/fault_injection.csv`, `monitoring.json` → `fault_injection` |
| Why each design choice | Ablation with paired cross-validation: rolling features, per-condition normalisation and drift-from-baseline each matter; a model ensemble did not help | `results/ablation/ablation_delta_vs_baseline.csv` |

**Two evaluation sets, never mixed:**
- **142 validation engines** (20% of each subset's training engines: 20 + 52 + 20 + 50), never used for training. They are replayed cycle by cycle to their recorded failure for the maintenance results: unplanned failures, life used and cost.
- **707 official NASA test engines** (100 + 259 + 100 + 248), used only to report RMSE (one prediction per engine at its last cycle).
- The linear and deep-model comparisons are 5-fold cross-validation on the training engines, with the same folds and the same RUL cap (125) as the boosting model.

Caveats we state openly: the replay uses the simulator's failure point as ground truth; the validation sets are small (20–52 engines per subset); cost ratios and lead times are assumptions, not real fleet data.
At a 15-cycle lead time the current 10-cycle safety buffer is too short for 1–2 engines. The buffer is a config value and should equal the real lead time.
The final run used the earlier promotion gate (challenger must be better with 80% bootstrap confidence; it promoted v2 on all four subsets). Across the 5 seeds that gate wrongly kept v1 in 2 runs, so the code now uses a non-inferiority gate (see `docs/methodology.md` §7.2).
See `docs/methodology.md` §11 and §14.

## Built vs planned

| Built and run | Planned (designed, not yet implemented) |
|---|---|
| Ingest + schema checks, causal cleaning, per-condition normalisation, past-only rolling features (leakage-tested) | Technician-outcome feedback loop (labels from completed work orders) |
| RUL model (gradient boosting, chosen by 5-fold CV by engine), calibrated 80% range, P(fail ≤ 30), anomaly model | Live streaming ingestion (MQTT/Kafka); today the stream is a cycle-by-cycle replay |
| Data-quality gate + CHECK_SENSOR tickets | Synthetic drift demo for the drift-triggered retrain path |
| Triage agent: severity ladder, due dates, human-review rules, hysteresis, review timeout | Prediction-distribution and operating-mix drift views in the dashboard |
| MLflow tracking + registry (champion / production aliases), PSI drift, retrain trigger, bootstrap promotion gate, rollback by alias | |
| Monitoring dashboard, fleet replay, policy and cost comparison, 5-seed robustness, ablation | |

## Repository layout

```
notebooks/
  pipeline_FD001.ipynb     the whole pipeline for one subset, section by section
  run_all_subsets.ipynb    runs FD001-FD004 end to end, writes the summary, dashboard and one zip
  ablation.ipynb           design ablation with paired cross-validation
  deep_models.ipynb        1D-CNN / GRU / TCN vs gradient boosting, same folds, 3 seeds per net
scripts/
  build_notebook.py        single source of all three notebooks (edit here, then regenerate)
  build_dashboard.py       builds the monitoring dashboard from a run's saved outputs
  cost_optimization.py     renewal-reward cost model, cost-optimal fixed interval, threshold tuning check
  policy_comparison.py     reactive vs fixed interval vs agent (standalone version)
  convert_to_csv.py        raw NASA text files -> headed CSVs
results/
  final_run/               per-subset outputs of the final run (JSON, CSV, plots) + dashboard.html
  seeds/                   5-seed robustness summaries
  gate_check/              rerun with the non-inferiority gate (seeds 42, 1, 3) + its dashboard
  ablation/                ablation tables
  deep/                    deep sequence models vs gradient boosting
docs/methodology.md        full design and evaluation write-up
index.html                 the dashboard (served by GitHub Pages)
```

## Reproduce

1. **Kaggle (what we used):** create a notebook from `notebooks/run_all_subsets.ipynb` and attach a C-MAPSS dataset. Set *Accelerator → GPU T4*, *Internet → On*, then *Save Version → Save & Run All*. About 20 minutes; output is `all_subsets_outputs.zip` with every result file and `dashboard.html`.
2. **Locally (CPU works, slower):** `pip install -r requirements.txt`, put the raw files in `DataSet/`, then run the notebook.
3. Cost model on a finished run: `python scripts/cost_optimization.py <run_dir>`. Dashboard: `python scripts/build_dashboard.py <run_dir>`.

Everything is open source: pandas, NumPy, scikit-learn, XGBoost, MLflow. The dashboard is plain HTML/SVG.
Data: NASA C-MAPSS (public domain), see `data/README.md`. Code: MIT licence, see `LICENSE`.
