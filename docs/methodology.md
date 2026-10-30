# Predictive Maintenance for a Connected Fleet

This document describes the complete solution in detail: the design, the reasoning behind each choice and the evidence from the runs.

**How to read the status labels**
- **[Built]** — implemented, run end to end on all four C-MAPSS subsets, numbers below come from that run.
- **[Designed]** — part of the architecture and specified here, not yet implemented. Planned next.
- All results are from **one run with a fixed seed** (42) on the official NASA test sets.
- **How the test sets were used:** no model was fitted on test data, and model choice, thresholds, calibration and promotion are decided on training and validation machines only. However, results from our *first* test runs did inform design iterations (the evaluation window, the headline metric, the CRITICAL rule, and the decision not to refit on all machines). Section 10 lists each of these changes and what triggered it.

---

## 1. The problem, restated

A fleet of connected machines (trucks, buses, compressors) streams sensor data. Today failures are found **after breakdown** or prevented by **fixed service intervals**: the first causes unplanned downtime, the second throws away useful life and still misses early failures. The brief asks for a **closed-loop system** that:

1. ingests sensor data (batch or streaming) and cleans it,
2. detects early signs of degradation (abnormal behaviour),
3. predicts **Remaining Useful Life (RUL)**: how many operating cycles a machine has left before failure,
4. **automatically raises a maintenance work order**, with **a human stepping in only for exceptions**,
5. tracks experiments and model versions, retrains when new data arrives, and offers a monitoring view.

Constraints: failures are rarer than normal readings (mild imbalance); the pipeline must run end to end with no manual steps; lightweight, **no LLM** (the agent uses rules/thresholds); open-source tools; design documented for review.

**Headline result.** Against today's practice on 142 held-out engines, our system had **0 unplanned failures while using ~90% of each machine's useful life**, versus ~62% for a fixed service schedule that is equally safe — and its maintenance cost rate is **22–47% below the cost-optimal fixed schedule** across failure-cost ratios of 3–20× (section 9.4).

**Our framing.** This is not "build the most accurate RUL regressor". It is a *decision system*: for every machine, every cycle, decide *do nothing / monitor / inspect / schedule maintenance / remove from service now*, and decide *when a human must look*. The ML models are inputs to that decision. We therefore judge the system mainly on decision quality (were machines near failure caught in time? how many false alarms? how many cases needed a human?) and use RUL error as a supporting metric.

**What a cycle is.** In C-MAPSS one row = one operating cycle (one flight). For a truck or bus a cycle maps to a trip or a shift; for a compressor, to an operating period. Every threshold below is in cycles and converts to calendar time through each asset's usage rate (e.g. a bus doing 2 shifts a day: 30 cycles ≈ 2 weeks).

---

## 2. The dataset and what we learned from it

NASA C-MAPSS turbofan engine degradation simulation. Each subset is a fleet of engines of the same type, each starting with unknown manufacturing variation and initial wear (normal, not a fault). Each row: engine id, cycle, 3 operating settings, 21 sensors. Training engines run until failure; test engines stop some time before failure, with the true remaining cycles given separately (`RUL_FD00x`).

| Subset | Train / test engines | Operating conditions | Fault modes | Train rows | Test RUL range | Test engines with RUL > 125 |
|---|---|---|---|---|---|---|
| FD001 | 100 / 100 | 1 (sea level) | 1 (HPC) | 20,631 | 7–145 | 11 |
| FD002 | 260 / 259 | 6 | 1 (HPC) | 53,759 | 6–194 | 57 |
| FD003 | 100 / 100 | 1 | 2 (HPC + fan) | 24,720 | 6–145 | 15 |
| FD004 | 249 / 248 | 6 | 2 (HPC + fan) | 61,249 | 6–195 | 67 |

Findings from our data analysis that shaped the design:

- **No missing values, duplicates or cycle gaps** in the provided files — but a real fleet will have all three, so the pipeline validates and handles them anyway.
- **7 of 21 sensors carry no information** in FD001 (constant or two-valued: s1, s5, s6, s10, s16, s18, s19). In FD003/FD004, s6 and s10 do vary (fan fault) and are kept.
- **In FD002/FD004 the operating condition dominates the raw signal.** 97–100% of each sensor's variance is explained by which of the six flight regimes the engine is in; raw sensors correlate with RUL at |r| ≤ 0.06. After normalising each sensor within its operating condition, the regime effect drops to 0% and, with smoothing, the correlation with RUL recovers to |r| ≈ 0.74–0.89 for 10 of 14 sensors (0.3–0.6 for the rest) — the same level as FD001.
- **Degradation is slow, then fast.** Sensor trends are flat for most of an engine's life and accelerate in roughly the last 70 cycles. A model cannot distinguish "250 cycles left" from "150 cycles left" — the standard response is a capped (piecewise-linear) RUL target (section 5.2).
- **Engines vary a lot.** Lifetimes range 128–543 cycles; each engine has its own baseline. Features relative to the engine's own early behaviour are more informative than absolute levels.
- **Sensor noise is substantial** (the generating paper describes layered process and measurement noise): point-to-point noise is about a third of the signal spread, so smoothing over a window is necessary.
- **Imbalance:** if "near failure" means RUL ≤ 30, it is 12.5–15% of rows.
- **The readme disagrees with the files** for FD004 engine counts (readme: 248/249; files: 249/248). We trust the files.

---

## 3. Architecture

### 3.1 End-to-end flow

```
                 ┌──────────────────────────── feedback (closed loop) ─────────────────────────────┐
                 │                                                                                  │
Sensor stream /  │                                                                                  │
batch files ──► 1 Ingest & validate ──► 2 Clean & quality gate ──► 3 Normalise per operating      │
                                                                     condition                      │
                                                                       │                            │
                                                                       ▼                            │
                                                         4 Causal rolling features                  │
                                                                       │                            │
                         ┌──────────────────┬──────────────────────────┼──────────────────────┐    │
                         ▼                  ▼                          ▼                      ▼    │
                 5a RUL regressor   5b Calibrated RUL      5c Early-warning         5d Anomaly     │
                 (gradient boost)      interval (10–90%)     classifier              detector      │
                                                             P(fail ≤ 30 cycles)    (Isolation F.) │
                         └──────────────────┴──────────────┬───────────┴──────────────────────┘    │
                                                           ▼                                        │
                                              6 Triage Agent (rules, no LLM)                        │
                                     severity · due date · human-review exceptions                  │
                                                           │                                        │
                              ┌────────────────────────────┼─────────────────────────┐              │
                              ▼                            ▼                         ▼              │
                      Work order OPEN            Work order NEEDS_REVIEW        Monitor / log       │
                      (automatic)                (human exception)              (no order)          │
                              └──────────────┬─────────────┘                                        │
                                             ▼                                                      │
                              7 Technician outcome  [Designed] ─────────────────────────────────────┘
                                             │
                                             ▼
   8 MLOps: experiment tracking · model registry (champion / production) · drift check · retrain
     trigger · champion–challenger promotion gate · monitoring view
```

### 3.2 Components and responsibilities

| # | Component | Responsibility | Status |
|---|---|---|---|
| 1 | Ingest & validate | Locate raw or CSV files, enforce schema, count missing / non-numeric / duplicate / gap issues, reject unusable input | [Built] |
| 2 | Clean & quality gate | Type coercion, de-duplication, per-machine gap filling with an `imputed` flag; sensor-health checks (dead, stuck, out of range); unseen-operating-condition check | [Built] |
| 3 | Condition normalisation | Detect operating regime, z-score each sensor within its regime, clip outliers | [Built] |
| 4 | Feature builder | Past-only rolling features per machine | [Built] |
| 5 | Model ensemble | RUL point estimate, calibrated interval, early-warning probability, anomaly score | [Built] |
| 6 | Triage Agent | Rule-based severity, due date, work order, human-review routing, per-machine order state | [Built] |
| 7 | Feedback | Technician outcome → labels → threshold recalibration and retraining | [Designed] |
| 8 | MLOps | MLflow tracking + registry, drift (PSI), retrain trigger, promotion gate, exports for monitoring | [Built] (incl. monitoring dashboard) |

### 3.3 Design principles
- **Decisions over accuracy:** the agent acts on a *calibrated lower bound* of remaining life, so uncertainty makes it act earlier, never later.
- **Leakage-safe by construction:** splits by machine, features use only past cycles (verified by a test), no model fitted or selected on test data (test results did inform early design iterations — section 10).
- **One code path for every subset:** a single config switch (`SUBSET`) and a per-subset table (condition normalisation on/off, number of fault modes). No hand-tuning per subset.
- **Fail safe and traceable:** every work order lists every rule that fired and every reason for human review; every model and threshold is versioned.

---

## 4. Data engineering

### 4.1 Ingestion and validation [Built]
- Reads either the raw space-separated `.txt` files (26 columns, no header) or converted CSVs; finds them automatically.
- Schema validation: required columns present, numeric types, positive integer unit/cycle; report of missing cells, non-numeric cells, duplicate (unit, cycle) rows, and cycle gaps per machine.
- Test labels are verified: every test engine's last-row RUL must equal the value in the RUL file (assertion).
- The same validation runs again on each new batch of data before retraining.

### 4.2 Cleaning [Built]
- Coerce to numeric, drop rows with no unit/cycle, de-duplicate (unit, cycle), sort by machine and time.
- **Missing values:** forward-fill **within each machine only** (sensor readings are continuous in time; filling across machines would leak). No back-fill — it would copy later cycles into earlier ones. Gaps at the very start of a machine's history become the training mean after normalisation. Every row with a missing value is flagged `imputed`; the agent sends work orders based on recently imputed data to human review.
- **Uninformative sensors:** dropped if they take ≤ 2 distinct values within an operating condition (automatic; FD001 keeps 14 sensors, FD003/FD004 keep 16).
- **Noise and outliers:** handled in two places — after per-condition normalisation (section 4.3) each sensor is clipped to its 0.1% / 99.9% training quantiles, and the rolling features (section 4.4) smooth the remaining noise.

### 4.3 Operating-condition normalisation [Built]
- Operating regime = rounded (op1, op2, op3). FD002/FD004 have exactly six regimes (altitude / Mach / throttle combinations); FD001/FD003 one.
- Each sensor is z-scored using the mean and standard deviation **of its own regime**, fitted on training machines only.
- Effect on FD002 (measured): variance explained by regime 0.97–1.00 → 0.000 for all 14 sensors; correlation with RUL from |r| ≤ 0.06 to 0.74–0.89 for 10 of 14 sensors after smoothing (FD001: 0.79–0.82 for the same sensors).
- This is the key step that lets the same pipeline handle mixed operating conditions — the stand-in for a real fleet where load, route, speed or ambient conditions vary.

### 4.4 Feature engineering [Built]
For each kept sensor, per machine, using **only the current and past cycles**:

| Feature | Window | Purpose |
|---|---|---|
| Normalised value | — | current level |
| Rolling mean | 10 cycles | smoothing noise |
| Rolling standard deviation | 10 cycles | instability / vibration-like spread |
| Rolling slope (least squares) | 20 cycles | trend — how fast it is degrading |
| Exponentially weighted mean | span 10 | smoothed level, reacts faster than a plain mean |
| Drift from own baseline | EWMA − mean of the machine's first 10 cycles | removes each machine's manufacturing / initial-wear offset |

Plus the cycle count (time in service). 85 features (FD001/FD002), 97 (FD003/FD004).

- Short histories are handled: windows shrink at the start of a machine's life (no missing values, no crash); machines with < 15 cycles of history are sent to human review by the agent.
- **Causality tests (run every time):** for randomly chosen machines, features are recomputed on the series truncated at a random cycle and compared to the full-series features at that cycle — they must be identical. A second test does the same for **cleaning + features** on machines with injected gaps (dense at the start of the series), so missing-value handling is proven past-only too. Together they show no future data leaks into any row, so scoring a row equals scoring a machine "live" at that cycle.

### 4.5 Labels [Built]
- Training RUL = last cycle − current cycle (machines run to failure).
- Test RUL = true final RUL (from the RUL file) + last observed cycle − current cycle.
- Regression target = **min(RUL, 125)** (section 5.2).
- Early-warning label = RUL ≤ 30 cycles.

### 4.6 Splits [Built]
All splits are **by machine**, never by row (rows of one machine are strongly correlated; row splits would leak):
- **Validation** 20% of machines — never trained on; used for checks and the promotion decision.
- **Initial batch** 50% — "the data we have at the start" → model v1.
- **New batch** 30% — "data that arrives later" → triggers the retraining flow → model v2.
- The **official test set** is only for reporting.

### 4.7 Data-quality gate [Built]
Flags only — it never changes the data. Everything is fitted on training machines and is past-only.
- **Unseen operating condition:** settings that match no training condition are mapped to the **nearest known condition** (not silent global statistics, which are ~80× off on FD002/FD004) and flagged; the agent sends the machine to human review. For single-condition subsets, settings outside the training range (plus a margin) are flagged.
- **Sensor health:**
  - *dead* — no reading at all;
  - *stuck* — identical for 15 cycles (window max − min = 0; a rolling standard deviation leaves rounding residue up to ~1e-5), checked only on sensors that **never** flatline in training — learned per subset, because some sensors are coarse by design (FD003's s6 and s10 sit still 28–96% of the time);
  - *out of range* — beyond the training min–max by half the range, for 2+ consecutive cycles. Skipped when the operating condition itself is unseen, so a new regime is not blamed on the sensors.
- **Result on the real data:** zero flags across all train and test data of all four subsets (no false sensor tickets on 707 test machines). Because the data has no broken sensors, section 9.8 injects faults to demonstrate the gate.

---

## 5. Machine learning

### 5.1 Models and why

| Model | Job | Why this choice |
|---|---|---|
| **Gradient-boosted trees** (XGBoost and scikit-learn HistGradientBoosting, chosen by cross-validation), Ridge regression as a baseline | Predict RUL | Engineered rolling features make this a tabular problem where boosted trees are strong, fast (seconds to train), robust to unscaled features and easy to explain. Ridge sets a floor: if trees didn't clearly beat it, the complexity wouldn't be justified. |
| **Quantile regression** (10% and 90%), conformally calibrated | Uncertainty interval around RUL | The agent needs to know *how sure* the prediction is: it acts on the lower bound and sends wide intervals to a human. |
| **Early-warning classifier** (gradient-boosted, `class_weight="balanced"`) | P(failure within 30 cycles) | A second, independent opinion near failure; disagreement with the RUL model is a human-review trigger. Directly addresses the imbalance requirement. |
| **Isolation Forest** trained only on each machine's early-life cycles (cycles 11–40) | Anomaly score: "this machine no longer behaves like it did when new" | Unsupervised — needs no failure labels, so it works on a new fleet from day one (a real fleet uses the first cycles after commissioning or maintenance). Cycles 11–40 are ≥ 88 cycles from failure in every training engine. Flags must persist 3 cycles to count (reduces noise alarms). |

**Why not an LSTM/CNN first?** Sequence models are popular on C-MAPSS and can reach strong scores, but they need more data per configuration, are harder to explain to a maintenance planner, slower to retrain in an MLOps loop, and the brief asks for a lightweight system. Our rolling-window features already give the trees the temporal information, and the trees clearly beat the linear baseline (CV capped RMSE ≈ 10–12 vs ≈ 17 for Ridge). We do not claim parity with published deep-learning results: many papers report RMSE against capped test labels, others against true labels, so comparisons need care; our true-label test RMSE is 11.7 / 23.1 / 12.5 / 23.7 (FD001–FD004). We then **tested this directly** (section 9.9): a 1D-CNN, a GRU and a TCN, each trained from 3 random starts, under the same folds and the same "clearly better" rule as the promotion gate. They lost to boosting by 3–6 cycles of capped RMSE on every subset, so boosting stays.

### 5.2 Key modelling decisions
- **RUL cap at 125.** Early in life sensors barely change, so the true RUL of a healthy machine is unknowable from data. Training on min(RUL, 125) — the standard piecewise-linear target for C-MAPSS — stops the model wasting capacity on the unpredictable healthy phase. Consequence: the model never predicts above 125. For an engine with 190 cycles left the error is ≥ 65 cycles, but it is on the safe side (early) and changes no maintenance decision.
- **Model selection and calibration by 5-fold cross-validation grouped by machine, per model version, on that version's own training machines only** (v1: the initial batch; v2: initial + new batch). Preprocessing is re-fitted inside every fold. No version's choices use data that "arrived" after it. Selection uses RMSE against the capped truth — the part of the error the model can actually reduce.
- **Evaluation rows that mirror the test set:** validation is scored on rows with ≥ 31 cycles observed and true RUL ≤ 200, matching how test machines are cut. Because features are causal, scoring a row equals scoring a truncated machine.

### 5.3 Handling imbalance (failures rarer than normal readings)
- Regression target capping removes the dominance of long healthy histories.
- Early-warning classifier trained with balanced class weights (positives 12.5–15%).
- Early-warning flag threshold chosen on out-of-fold predictions to **maximise F2** (recall weighted 2× over precision — a missed failure costs more than an extra inspection). This recall-first flag is reported for monitoring; the agent's HIGH rule deliberately uses P ≥ 0.5 instead (section 6.2).
- Evaluated with **PR-AUC** (not just accuracy / ROC-AUC, which flatter imbalanced problems).
- The anomaly detector sidesteps imbalance entirely: it learns only early-life ("as new") behaviour.

### 5.4 Uncertainty calibration
Raw quantile models were over-confident: their 10–90% band covered only 55–70% of held-out cases instead of 80%. We apply **conformalized quantile regression**: on out-of-fold cross-validation predictions, find the margin (in cycles) that makes the widened band cover 80% of held-out rows with true RUL ≤ 125, and add it to every interval. Margins: 4.3 / 2.5 / 5.2 / 2.5 cycles (FD001–FD004). Result on the official test sets: coverage 0.84–0.85 (target 0.80). Calibrating only on rows ≤ 125 matters: above the cap the truth (capped at 125) sits at the clipped upper bound and is "covered" automatically, which would shrink the margin exactly where decisions are made.

### 5.5 Metrics we report and why
- **RMSE against the true RUL** — the standard C-MAPSS number.
- **RMSE against the capped truth** — the error the model can control. Both are headline numbers, side by side.
- **NASA PHM08 score** — asymmetric, exponential penalty: `exp(−d/13) − 1` for early and `exp(d/10) − 1` for late predictions (d = predicted − true). Lower is better; late predictions (unplanned failures) cost more.
- **Error by true-RUL band** — shows where the error comes from.
- **Interval coverage**, **classifier PR-AUC / recall / precision**, **anomaly false-alarm rate and lead time**.
- **Decision metrics** (section 9) — the ones that matter most for the business.

---

## 6. Triage Agent (rule-based, no LLM) [Built]

### 6.1 Inputs per machine, per cycle
Predicted RUL, calibrated lower and upper bound, P(fail within 30), persistent-anomaly flag and score, cycles of history observed, recent-imputation flag.

### 6.2 Severity rules
All rules are evaluated; the **highest severity that fires wins**, and **every rule that fired is recorded** on the work order as a reason.

| Severity | Fires when (any) | Recommended action | Work order |
|---|---|---|---|
| **CRITICAL** | calibrated **lower-bound** RUL ≤ 15 | Remove from service now; inspect and overhaul the affected module before the next operation | P1, due now |
| **HIGH** | predicted RUL ≤ 30, or P(fail ≤ 30) ≥ 0.5 | Schedule maintenance before the due cycle; reserve parts and a slot | P2, due within (lower bound − 10 cycles) |
| **MEDIUM** | predicted RUL ≤ 60, or persistent anomaly **confirmed** by the RUL model (predicted RUL ≤ 100) | Plan an inspection; increase monitoring frequency | P3, due within (lower bound − 10 cycles) |
| **LOW** | predicted RUL ≤ 100, or a persistent anomaly the RUL model does not confirm (watch) | Monitor | none (logged) |
| **NORMAL** | otherwise | — | none |

Design choices worth noting:
- **CRITICAL uses the lower bound**, so a less certain prediction escalates *earlier* (safety first).
- **An anomaly alone does not raise a work order.** The early-life anomaly detector fires earlier than the RUL model (median 42–70 cycles before failure), and on its own it produced most false alarms. Decided on **validation** engines (every test-like cycle scored): requiring the RUL model to agree (RUL ≤ 100) cut work orders on healthy cycles (true RUL > 100) by 2–12× per subset (e.g. FD003 5.5% → 0.45%, FD001 6.0% → 1.8%), with **identical** detection of machines within 60 and within 30 cycles of failure. An unconfirmed anomaly is logged as a LOW watch item.
- **HIGH uses P ≥ 0.5, not the F2 threshold.** HIGH should mean "more likely than not to fail within 30 cycles"; the recall-first F2 threshold (≈ 0.05–0.16) is used for the early-warning flag. The agent's recall comes from the RUL tiers and MEDIUM.
- **CRITICAL uses only RUL, not the classifier.** In our first run a rule "P(fail ≤ 30) ≥ 0.8 → CRITICAL" turned 13 of 15 true-HIGH engines into CRITICAL: the classifier answers "within 30 cycles?" and is ≈ 1.0 for every engine under 30, so it cannot separate 15 from 28 cycles. Each model now drives only the decision it was trained for.
- **Due date** = max(0, ⌊lower bound⌋ − 10): a 10-cycle safety buffer, never negative.
- **Recommended action depends on the fault modes of the asset type** (HPC module; HPC and fan modules for FD003/FD004).

### 6.3 Human only for exceptions
A work order gets status **NEEDS_REVIEW** instead of **OPEN** when any of these hold:
1. **Uncertain prediction:** calibrated interval wider than the threshold learned in cross-validation (the 90th percentile of interval width among likely work-order cases, so ~10% of cases by design). Learned values: 33.6 / 32.0 / 44.7 / 44.9 cycles (FD001–FD004).
2. **Models disagree:** P(fail ≤ 30) ≥ 0.5 but predicted RUL > 80, or predicted RUL ≤ 30 but P(fail ≤ 30) < 0.1.
3. **Too little history:** fewer than 15 cycles observed.
4. **Data quality:** recent sensor values were imputed.
5. **Unknown operating condition:** the machine is running at settings never seen in training (nearest known condition used for normalisation).
6. **Sensor fault:** a dead, stuck or out-of-range sensor (section 4.7). The sensor itself also gets a separate `CHECK_SENSOR` ticket.

**CRITICAL orders are always dispatched (OPEN) even when a review reason exists** — the review is advisory; safety does not wait for a human. Review rate observed: 9.5–17.6% of work orders. Note this is largely **design-controlled**: the uncertainty threshold is set at the 90th percentile, so ~10% is a chosen review capacity, not an independent result.

### 6.4 Agent state (per machine)
- One open order per machine; no duplicates while severity is unchanged.
- When severity **escalates**, a new order is raised and the old one is marked SUPERSEDED (full audit trail).
- **Hysteresis:** an order is downgraded (or closed, when no order is needed any more) only after 5 consecutive calmer cycles, so one noisy cycle never cancels it.
- **Review timeout:** a NEEDS_REVIEW order nobody handles within 10 cycles is flagged `review_overdue` (escalate to a supervisor).
- **Sensor tickets:** one `CHECK_SENSOR` ticket per machine, re-raised only when another sensor joins the fault list.
- **Data-trust re-check:** if a sensor fault, an unseen operating condition or filled-in data appears on a machine whose order is *already open*, that order switches to NEEDS_REVIEW with the new reason and the cycle it was requested (CRITICAL stays dispatched, flagged for review). Ordinary changes in uncertainty don't re-open review — they were judged when the order was raised. Tested on scripted scenarios; on the real (clean) data it never triggers, and the fleet replay is identical to the previous run.

### 6.5 Work order format (real examples from the run)
Automatic CRITICAL order (FD004):
```json
{
  "work_order_id": "WO-FD004-008-0248", "created_at": "2026-10-27T17:36:06+00:00",
  "severity": "CRITICAL", "priority": "P1", "status": "OPEN",
  "unit": 8, "at_cycle": 248, "predicted_rul": 13.9, "rul_lower": 9.8, "rul_upper": 33.4,
  "p_fail_window": 0.999, "anomaly": true, "due_within_cycles": 0,
  "recommended_action": "Remove from service now; inspect and overhaul the HPC and fan modules before the next operation.",
  "reasons": ["lower-bound RUL 10 <= 15", "predicted RUL 14 <= 30", "P(fail within 30) 1.00 >= 0.5",
              "predicted RUL 14 <= 60", "persistent anomaly (score 0.694)", "predicted RUL 14 <= 100"],
  "review_reasons": [], "model_version": "rul_fd004/v2", "agent_version": "rules-1.0"
}
```
Human-exception order (FD002) — the anomaly detector flags the engine while the RUL model still says ~82 cycles and the interval is wide:
```json
{
  "work_order_id": "WO-FD002-008-0173", "created_at": "2026-10-27T17:29:04+00:00",
  "severity": "MEDIUM", "priority": "P3", "status": "NEEDS_REVIEW",
  "predicted_rul": 82.4, "rul_lower": 65.8, "rul_upper": 99.7, "p_fail_window": 0.0, "anomaly": true,
  "due_within_cycles": 55,
  "recommended_action": "Plan an inspection of the HPC module and increase monitoring frequency.",
  "reasons": ["persistent anomaly (score 0.656)", "predicted RUL 82 <= 100"],
  "review_reasons": ["uncertain: interval 66-100 wider than 32.0"]
}
```

### 6.6 Streaming behaviour
Every validation engine (never trained on, all run to failure — 142 in total) is streamed through one stateful agent, cycle by cycle, like a live fleet. FD001 example engine: MEDIUM order raised with **72** true cycles left → escalated to HIGH at **36** → CRITICAL at **23**.

Fleet-wide (FD001 / FD002 / FD003 / FD004):

| | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Median true cycles left at the first work order | 70 | 67 | 62.5 | 61 |
| Median true cycles left at the first HIGH+ order | 32.5 | 32 | 31 | 31 |
| Engines reaching CRITICAL before failure | 100% | 100% | 100% | 100% |
| Orders per engine (created + escalated + downgraded) | 3.4 | 3.3 | 3.1 | 3.3 |
| Engines with a downgrade or closure (hysteresis at work) | 7 / 20 | 10 / 52 | 2 / 20 | 8 / 50 |
| Fewest cycles left at planned service | 16 | 15 | 14 | 12 |

---

## 7. MLOps

### 7.1 Experiment tracking and model registry [Built]
- **MLflow** (SQLite backend): every cross-validation candidate, every trained version, the promotion decision, the evaluation and the agent run are logged with parameters, metrics and artifacts.
- **Model registry** per subset (`rul_fd001` … `rul_fd004`) with aliases **`champion`** (best validated model) and **`production`** (what the agent uses).
- The full model **bundle** (preprocessing statistics, feature list, all four models, thresholds, calibration margin) is saved and versioned together, so a model and its thresholds can never get out of sync.
- If MLflow is unavailable, a JSON tracker with the same interface takes over — the pipeline never stops on a tracking problem.

### 7.2 Retraining flow when new data arrives [Built]
1. **v1**: selection and calibration on the initial batch only → trained → registered → `champion`.
2. **New batch arrives** → schema validation → **drift check**: Population Stability Index (PSI) per sensor, new vs reference data.
3. **Retrain trigger:** new machines ≥ 20% of the current training set, **or** any sensor PSI > 0.2.
4. **v2**: selection and calibration re-run on initial + new (model type, thresholds, interval margin and review threshold all recalibrated as one bundle) → trained → registered.
5. **Promotion gate (non-inferiority):** v2 is the same design trained on more data, so it becomes champion **unless it is clearly worse**, judged on 2,000 bootstrap resamples of the validation machines (only 20–52 machines, so single-split differences are noisy). v1 stays if, in at least 80% of resamples, v2's capped RMSE is worse by more than 0.5 cycles (a margin below the 0.6–1.2-cycle fold-to-fold spread), or its capped NASA score is more than 10% worse. Ties and negligible differences count as "not worse". A stricter *superiority* test (v2 must be clearly better) remains available for genuine design changes. The test set is never used here.
   - *Why it changed:* the earlier superiority test, run across 5 data splits, rejected v2 in 3 of 20 runs, and in 2 of them v2 was actually better on test (section 9.7). On v2-vs-v1 differences across those 20 runs, v2 was never clearly worse.
   - *Checked on real predictions:* identical models → promote; +0.1-cycle noise → promote; ~1 cycle worse → block; clearly late → block; better → promote.
   - Seed-42 results (the reported ones) are unchanged: v2 promoted on all four subsets under either test (it was better in 92% / 100% / 98% / 100% of resamples).
   - *Confirmed on Kaggle* (29 Oct, seeds 42, 1 and 3, 12 runs): **v2 promoted in all 12**, including the three runs the old test blocked: FD001 seed 3 (test capped RMSE 10.11 vs v1's 11.16), FD004 seed 1 (11.17 vs 11.36) and FD004 seed 3 (11.41 vs 12.47; its NASA criterion passed with P = 0.59, below the 0.8 block level). 0 unplanned failures in all 12.

Observed: on all four subsets the trigger fired on data volume (new batch = 60% of the training set); drift was negligible (max PSI 0.014–0.05 — expected, since both batches come from the same simulated fleet). v2 was promoted on all four; validation capped RMSE improved by 0.5–1.0 cycles. Because selection now runs per version, the chosen model type can change at retraining — on FD001, v1 (50 engines) selected HistGradientBoosting and v2 (80 engines) selected XGBoost.

**Rollback:** `production` is an alias in the registry, so rolling back a bad production model means repointing the alias to the previous version — no retraining, and the old bundle (model + thresholds) is restored as a unit. This has not yet been exercised in a run.

**What the runs do and don't prove:** the volume trigger fires by construction (the new batch is 60% of the training set), and the drift path has never fired. The gate's blocking behaviour was exercised on constructed cases (a worse challenger is blocked), and the earlier superiority test did block v2 in 3 of 20 seed runs. A synthetic-drift demonstration is planned (section 11.2).

An experiment we ran and **rejected**: retraining the champion design on *all* training machines (including validation) for production. On FD003 it changed test RMSE from 12.45 to 12.70 and NASA from 273 to 267 — no measurable gain. This judgement was made on test results, so we kept the simpler policy (production = the validated champion) for all subsets rather than adopting the refit.

### 7.3 Monitoring [Built]
Exported per run: `monitoring.json` (cross-validation results, version history, drift per sensor, promotion decision, champion vs production, error by RUL band, agent evaluation, configs, data splits), predictions, work orders, plots (true vs predicted, error distribution, machine trajectories, classifier precision/recall, streaming replay).
**[Built] Monitoring dashboard** (`dashboard.html`, generated automatically at the end of `run_all_subsets.ipynb` by `scripts/build_dashboard.py`; one offline HTML page, no server): cross-subset overview; fleet table ranked by urgency with per-engine predicted RUL, 80% range and actual RUL; work-order queue (open / needs review) with full order cards; a live replay that streams the validation engines cycle by cycle and shows the agent's orders being created, escalated, downgraded and closed; model version history, promotion decision and bootstrap figures; drift (PSI) per sensor; CV model comparison; error and coverage by RUL band; ingest checks and fault-injection results; agent confusion matrix, policy comparison and cost table. **[Designed]** additions: prediction-distribution drift and operating-condition mix over time.

### 7.4 Automation and reproducibility [Built]
- **One notebook runs all four subsets end to end** with no manual steps: fresh namespace per subset (like restarting the kernel), failure isolation (one subset failing does not stop the others), clean start (old artifacts wiped), one combined output with a cross-subset summary. Total runtime ≈ 19 minutes on a Kaggle GPU session (4.0–5.9 min per subset, including two cross-validation passes per subset — one per model version — the fleet replay and the fault-injection demo).
- Fixed seed; deterministic training settings. Two Kaggle sessions on the same kind of machine gave identical results in all 12 runs compared; on a different Kaggle machine (same T4 GPU type, same library versions) GPU XGBoost differed in the last digits, which can flip the near-tie between HistGB and XGBoost (section 9.7: decisions identical, capped RMSE within ±0.3). Schema validation, label checks, split-disjointness assertion and the causality test run every time.
- Training runs on Kaggle; everything is open source (pandas, NumPy, scikit-learn, XGBoost, MLflow; the dashboard is plain HTML/SVG).

---

## 8. Dataset coverage: one design, four levels of difficulty

| Step up in difficulty | What changes in the data | Component that handles it | Effect |
|---|---|---|---|
| FD001 (baseline) | 1 condition, 1 fault | core pipeline | capped RMSE 10.55 |
| FD002 | 6 operating conditions | automatic regime detection + per-condition normalisation | regime variance 97–100% → 0%; capped RMSE 10.81 |
| FD003 | 2 fault modes (HPC + fan) | automatic sensor selection keeps fan-relevant sensors (s6, s10); calibration widens intervals; agent names both modules | capped RMSE 10.85; wider review threshold (44.7 vs 33.6) |
| FD004 | 6 conditions + 2 faults | all of the above, unchanged code | capped RMSE 10.96 |

The only per-subset input is a two-field table (normalise by condition? how many fault modes?). Everything else — sensor selection, calibration margin, alert threshold, review threshold, model choice — is learned from the data by the same code.

---

## 9. Results (official test sets, final run)

Model numbers come from the final Kaggle run of the all-subsets notebook. Agent numbers (9.3) come from the final agent rules (including the anomaly-confirmation rule, section 6.2) applied to that run's saved predictions — the agent is deterministic rules, so a notebook rerun reproduces them exactly.

### 9.1 Headline table

| | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| Model chosen by CV | XGBoost | XGBoost | HistGB | HistGB |
| CV RMSE (capped) | 10.19 | 11.32 | 10.65 | 12.18 |
| **Test RMSE vs capped truth** | **10.55** | **10.81** | **10.85** | **10.96** |
| Test RMSE vs true RUL | 11.66 | 23.12 | 12.45 | 23.68 |
| Test MAE | 8.53 | 14.77 | 8.97 | 15.79 |
| Bias (predicted − true) | −0.8 | −7.1 | +0.4 | −7.6 |
| NASA score (sum, lower is better) | 205 | 4,196 | 273 | 3,238 |
| Interval coverage (target 0.80) | 0.85 | 0.85 | 0.85 | 0.84 |
| Ridge baseline, CV RMSE (capped) | 17.39 | 16.88 | 17.30 | 17.04 |

### 9.2 Why true RMSE doubles on FD002/FD004
FD002/FD004 test sets contain many very healthy engines (true RUL up to 195; 57 and 67 engines above the 125 cap, vs 11 and 15 in FD001/FD003). Error by true-RUL band, FD004:

| True RUL | Engines | RMSE | Share of squared error |
|---|---|---|---|
| ≤ 30 | 53 | 4.5 | 1% |
| 30–60 | 38 | 10.6 | 3% |
| 60–100 | 57 | 17.2 | 12% |
| 100–125 | 33 | 10.1 | 2% |
| 125–150 | 25 | 20.2 | 7% |
| 150–195 | 42 | 49.6 | **74%** |

FD002 is the same pattern (73% of squared error from the 150–194 band). These errors are all **early** predictions on healthy engines — no maintenance decision depends on them. Within 30 cycles of failure, RMSE is 3.0–4.5 cycles on all subsets.

### 9.3 Decision quality (the triage agent on the whole test fleet)

| | FD001 | FD002 | FD003 | FD004 | Total |
|---|---|---|---|---|---|
| Test engines | 100 | 259 | 100 | 248 | 707 |
| Work orders raised | 42 | 119 | 44 | 100 | 305 |
| **Engines within 30 cycles of failure rated HIGH or CRITICAL** | 24/25 | 61/61 | 19/20 | 53/53 | **157/159** |
| Engines within 30 cycles of failure that got any work order (MEDIUM or above) | 25/25 | 61/61 | 20/20 | 53/53 | 159/159 |
| True-CRITICAL (≤ 15) classed CRITICAL | 10/10 | 45/46 | 10/10 | 36/36 | 101/102 |
| Exact severity match | 71% | 77% | 77% | 79% | |
| Orders sent to human review | 9.5% | 17.6% | 13.6% | 12.0% | |
| False alarms (order raised, true RUL > 100) | 1 | 2 | 0 | 2 | 5 |

"True severity" is defined by us from the true RUL with the same bands the agent uses (≤ 15 CRITICAL, ≤ 30 HIGH, ≤ 60 MEDIUM, ≤ 100 LOW). Mismatches are almost always one level off and mostly on the cautious side (4–8 true-HIGH engines per subset escalated to CRITICAL). The one true-CRITICAL engine labelled HIGH (FD002) still received a work order.

FD004 severity matrix (rows = true, columns = agent):

| | NORMAL | LOW | MEDIUM | HIGH | CRITICAL |
|---|---|---|---|---|---|
| NORMAL | 87 | 11 | 2 | 0 | 0 |
| LOW | 15 | 32 | 9 | 1 | 0 |
| MEDIUM | 0 | 3 | 29 | 6 | 0 |
| HIGH | 0 | 0 | 0 | 11 | 6 |
| CRITICAL | 0 | 0 | 0 | 0 | 36 |

### 9.4 Versus today's practice: maintenance-policy comparison
The problem statement names today's practice: run to breakdown, or service on fixed intervals. We compare both against our system on the **142 validation engines** (20 / 52 / 20 / 50 per subset; never trained on by the champion, all run to failure; no test data used). Script: `scripts/policy_comparison.py`, using the exact agent code and the run's saved predictions.

- **Reactive:** run to failure — every failure is unplanned.
- **Fixed interval T:** service every engine at cycle T, with T set from the *non-validation* training engines' history, as a fleet would set it: the largest T with no failures in that history (T = 127 / 127 / 144 / 127), and the T allowing 5% failures.
- **Ours:** replay the agent cycle by cycle; service at the due cycle of the first HIGH or CRITICAL work order. If the engine's life ends first, it counts as an unplanned failure.

| Policy | Unplanned failures | Useful life used | Cycles wasted per engine |
|---|---|---|---|
| Reactive (run to failure) | 142 / 142 | 100% | 0 |
| Fixed interval, 0% failures in history | 0 / 142 | 55.6–67.6% | 66–123 |
| Fixed interval, 5% failures in history | 9 / 142 | 67.6–77.8% | 46–96 |
| **Predictive agent (ours)** | **0 / 142** | **88.7–90.6%** | **21–27** |

Per subset, life used by ours vs the equally safe fixed interval: FD001 88.7 vs 67.6% · FD002 89.1 vs 64.9% · FD003 90.6 vs 66.0% · FD004 89.5 vs 55.6%.

- **Safety margin:** every engine was serviced before failure, at least 12 cycles early (median 21). One FD004 engine was serviced 217 cycles early (a costly false call, included in the waste figures).
- **The reverse view:** a fixed interval using as much life as ours (T ≈ 180–240) would let **68 of 142 engines (48%)** fail in service.
- **Caveats:** one run; small validation sets; service assumed to happen exactly at the due date; "useful life" measured against the simulator's failure point; no repair lead times or costs modelled.

**Cost view (operations research — age-replacement / renewal-reward).** `scripts/cost_optimization.py`, on the same validation engines and saved predictions; the agent's service rule is reproduced exactly (0 mismatches with the real agent on all 142 engines).
- **Cost model:** a planned service costs 1, an unplanned failure costs *r* (3×, 5×, 10×, 20×, since real ratios are unknown); cost rate = total cost / total cycles operated, so failures and early service (life thrown away) are both penalised. A **lead time** of 0 / 10 / 15 cycles is added: a service planned fewer than that many cycles before failure can't be carried out and counts as a failure.
- **Strongest baseline:** the **cost-optimal fixed interval T\*** (the classic age-replacement policy), chosen on the non-validation engines' history for each *r* and lead time, then scored on the validation engines.
- **Result:** for lead times of 0–10 cycles and every *r* from 3 to 20, the agent's cost rate is **22–47% lower than the cost-optimal fixed schedule, with 0 unplanned failures** (e.g. *r* = 10, no lead time: FD001 5.85 vs 7.87, FD004 4.48 vs 7.87 per 1,000 cycles).
- **Should the thresholds be cost-tuned?** Tested with 2-fold cross-validation over the validation engines (grid over the HIGH threshold, CRITICAL lower bound and safety buffer). Without a lead time the optimiser runs to the most aggressive corner (median margin at service ~10 cycles instead of ~21, one engine at 4) because none of these engines happened to fail. With a 10-cycle lead time the tuned thresholds caused out-of-fold failures (FD002: 1, FD004: 2) and usually cost more. **Conclusion: on 20–52 engines per subset, tuning doesn't generalise; the current thresholds are kept** as the more robust choice — they are near cost-optimal and hold zero failures up to a 10-cycle lead time.
- **Design guidance:** the safety buffer should equal the real maintenance lead time. At a 15-cycle lead time the 10-cycle buffer is too thin for 1–2 engines (FD003, FD004), and at *r* = 20 the fixed schedule then wins on FD003. In a real fleet the buffer is set from actual parts and scheduling lead times (it is a config value).

### 9.5 Supporting models and calibration (validation machines unless stated)
- Early-warning classifier: PR-AUC 0.96–0.99, recall 0.97–0.99, precision 0.85–0.89.
- **Interval coverage by true-RUL band (test, target 80%):** ≤ 30 cycles 89–100% (true value below the lower bound in 0–11% of engines — the band CRITICAL relies on); 30–60 cycles 81–95%; 60–125 cycles 72–75% (under target, see 11.2).
- **Confidence on the catch rate:** "engines within 30 cycles rated HIGH or CRITICAL" has 95% lower bounds of 0.81 / 0.94 / 0.76 / 0.93 (FD001–FD004) — small fleets (20–61 urgent engines) limit how strong the claim can be.
- Anomaly detector (trained on early-life cycles 11–40): persistent alarms on 0.8–4.9% of healthy cycles (per cycle; "healthy" = true RUL > 125, an evaluation proxy); detects 95–100% of cycles in the last 30 before failure; flags every validation engine; median first alarm 42–70 cycles before failure (earlier than the previous RUL-labelled version's 42–53, at the cost of more alarms — which the agent's confirmation rule absorbs, section 6.2).

### 9.6 Ablation study: what each design choice is worth
`notebooks/ablation.ipynb`. One design choice changed at a time; every variant scored with the **same 5-fold cross-validation on the non-validation training engines** (paired folds, no test data), HistGradientBoosting as the base model. Metric: capped RMSE on held-out test-like rows; "Δ" = change vs the current design (+ = worse); "folds" = in how many of the 5 folds the variant beat the current design.

| Change | FD001 Δ | FD002 Δ | FD003 Δ | FD004 Δ | Folds better | Verdict |
|---|---|---|---|---|---|---|
| Raw sensors only (no rolling features) | +8.9 | +7.1 | +6.4 | +6.4 | 0/5 everywhere | **Essential** — the engineered features are the biggest single gain |
| No per-condition normalisation | — | +6.9 | — | +6.8 | 0/5 | **Essential** on multi-condition data |
| Ridge instead of gradient boosting | +7.2 | +5.3 | +6.6 | +4.9 | 0/5 | **Essential** — non-linearity matters |
| No drift-from-own-baseline feature | +5.5 | +5.2 | +3.3 | +4.5 | 0/5 | **Essential** — each engine's own baseline carries real signal |
| No cycle count | +4.1 | +3.3 | +2.0 | +3.6 | 0/5 | **Essential** |
| No trend (slope) features | +0.01 | +0.06 | −0.07 | −0.04 | 1–3/5 | **No measurable value** — could be removed (≈ 14–16 fewer features) |
| Shorter windows (5/10/5) | +0.1 | +0.1 | −0.3 | +0.1 | 1–4/5 | Within noise (only FD003 prefers shorter) |
| Longer windows (30/50/30) | +0.4 | +0.1 | +0.8 | +0.2 | 0–2/5 | Slightly worse — current 10/20 windows are a good choice |
| XGBoost instead of HistGB | +0.0 | −0.2 | +0.1 | −0.1 | 1–5/5 | Within noise — the library choice doesn't matter |
| **HistGB + XGBoost average** | **−0.10** | **−0.17** | **−0.03** | **−0.11** | **4/5 · 5/5 · 2/5 · 4/5** | **Small, fairly consistent gain (~1%)** — a candidate improvement |

**The RUL cap** needs a fairer comparison, because the table's metric is itself capped at 125 (which favours training at 125). On true RMSE and on the decision zone (true RUL ≤ 60):
- **Cap 140:** true RMSE **2.5–3.5 cycles better** (e.g. FD001 20.0 vs 22.8, FD004 23.2 vs 26.9), decision-zone error about the same (+0.04 to +0.16 cycles), NASA score worse (more late predictions on healthy engines).
- **Cap 110:** decision-zone error slightly better (−0.2 to −0.4 cycles) but true RMSE much worse (+4.7 to +5.4).
- **No cap:** similar true RMSE but the decision zone gets worse and the NASA score explodes (many late predictions).
- **Conclusion:** 125 is a sound middle choice for decisions; 140 would improve the headline true RMSE at almost no decision cost — a trade-off worth revisiting, not a clear win.

**Calibration (cross-fitted: each fold's margin comes from the other folds):** raw 10–90% quantile ranges cover 57–71% of held-out cases; after conformal calibration 80.0–80.3% (target 80%). In the RUL ≤ 30 band that CRITICAL relies on: 61–74% raw → **91–96% calibrated**.

**What this means:** the big design choices (rolling features, per-condition normalisation, each engine's own baseline, cycle count, non-linear model, calibration) each carry several cycles of error and win in every fold; the small ones (slope features, window lengths, XGBoost vs HistGB) are within noise. Averaging HistGB and XGBoost gains ~1% (better in 15 of 20 folds) — too small to justify running two models, so it was not adopted; dropping the slope features would simplify the model at no cost.

### 9.7 Robustness across 5 data splits
The full pipeline was rerun with 5 seeds (42, 1, 2, 3, 4); each seed changes which engines are validation / initial / new, the CV folds and the model randomness. Seed 42 reproduces the reported results exactly.

**Reproducibility, and its limit.** A second Kaggle session (29 Oct, seeds 42, 1 and 3, same code) reproduced all 12 subset × seed runs **identically** to the earlier one, to every decimal. A third session (29 Oct, the non-inferiority gate) landed on a different Kaggle machine — same T4 GPU type and library versions (Python 3.13.15, scikit-learn 1.6.1, XGBoost 3.4.1), but ~40% slower even in CPU-only steps. There, HistGB and Ridge were again bit-identical, but **GPU XGBoost differed in the last digits** (e.g. FD001 v2 CV 10.194 vs 10.240). HistGB and XGBoost are within ~0.05 cycles in CV, so this flipped the selected model on FD001 (XGBoost → HistGB) and FD004 (HistGB → XGBoost): seed-42 capped test RMSE 10.23 / 10.89 / 10.85 / 11.22 instead of 10.55 / 10.81 / 10.85 / 10.96 (mean 10.80 vs 10.79). **Every decision metric was identical:** 0 failures, 89.4% vs 62.2% of life used, 157/159 near-failure engines at HIGH or above, 5 false alarms, same coverage. So decisions reproduce exactly; capped RMSE reproduces within ±0.3 across Kaggle machines. Fix (designed, section 11.2): XGBoost on CPU with a fixed thread count, and prefer HistGB unless XGBoost is clearly better.

| Claim | Seed 42 | Range across 5 seeds (20 runs) |
|---|---|---|
| Unplanned failures on validation engines | 0 | **0 in all 20 runs** |
| Useful life used: ours vs equally safe fixed schedule (all 142 engines) | 89.4% vs 62.2% | 89.2–89.8% vs 61.8–64.5% |
| Near-failure engines with no work order | 0 | **0 in all 20 runs** |
| Near-failure engines rated HIGH or CRITICAL | 95–100% | 95–100% |
| Cost vs cost-optimal fixed schedule (failure 3–20×, lead time 0–10) | 22–47% lower | **17–71% lower (median 36%)**, 0 failures |
| Capped test RMSE | 10.55 / 10.81 / 10.85 / 10.96 | FD001 10.5–11.2 · FD002 10.8–11.2 · FD003 10.6–10.9 · FD004 11.0–12.5 |
| False alarms per subset | 0–2 | 0–3 |
| Interval coverage (target 0.80) | 0.84–0.85 | 0.78–0.87 |
| Review rate | 9.5–17.6% | 7.3–22.2% |

**What the extra splits revealed — the promotion gate is too strict.** It rejected v2 in 3 of 20 runs: FD004 seed 1 (v2 worse on validation; equal on test — correct), FD001 seed 3 (v2 better in only 72% of resamples; but better on test, 10.02 vs 11.16 — wrong call) and FD004 seed 3 (v2 better in 99% of resamples but its validation NASA score 13% worse than the 10% limit; better on test, 11.41 vs 12.47 — wrong call). The gate tested for *superiority*, but our retrain is the same design on 60% more data, where the right test is *non-inferiority* ("promote unless clearly worse"). **Changed** (section 7.2): non-inferiority with a 0.5-cycle margin for same-design retrains; superiority kept as an option for design changes. Seed 42 is unaffected (all four subsets promote either way). **Confirmed on Kaggle** with seeds 42, 1 and 3: v2 promoted in all 12 runs, including the three the old test blocked (section 7.2).

### 9.8 Fault-injection demonstration (sensor faults and unseen conditions)
The real data has no broken sensors, so for each subset four real test engines are broken over their last cycles and run through the same pipeline: a sensor **stuck** for 25 cycles, a sensor **out of range** for 5 cycles, a sensor that **dies** for its last 20 cycles (missing values get forward-filled, so it then looks stuck), and a switch to an **operating condition never seen** in training for 10 cycles.

- **Detected: 16 / 16.** Out-of-range in 1 cycle; unseen condition immediately; stuck/dead within the 15-cycle window on single-condition subsets, and within 1–6 cycles on six-condition subsets (a frozen value soon looks out of range when the condition changes).
- **Tickets:** exactly one `CHECK_SENSOR` ticket per broken sensor; none for the unseen condition (it isn't the sensors' fault); zero tickets on the clean test fleets.
- **Why it matters — bad data hides alarms:** on FD002 a stuck s11 made the model think an engine with **14.8** cycles left had **69**, dropping it from CRITICAL to HIGH; on FD004 a stuck s11 moved another engine from 89 to 109 cycles (LOW to NORMAL). The gate caught both and raised the sensor tickets; on FD002 it also sent the open order to human review (the FD004 engine had no order to review).
- **What it also showed:** an unseen operating condition pushed one FD004 engine from 51 to 10 cycles (MEDIUM to CRITICAL), dispatched by the safety-first rule while flagged for review (listed in 11.2). The run also showed that orders already open before a fault appeared kept status OPEN; the agent now re-checks open orders for data-trust problems (section 6.4).

### 9.9 Deep sequence models vs gradient boosting (`deep_models.ipynb`)
**Question:** do models built for temporal data beat our boosted trees? Three standard C-MAPSS architectures were trained on windows of the last 30 cycles of per-condition-normalised sensors plus `cycle` (past only; a causality test checks it): a **1D-CNN**, a two-layer **GRU** and a **TCN** (causal dilated convolutions). Same cleaning, normalisation, RUL cap and engine splits as the pipeline; **the same 5 paired CV folds** as the ablation; nets early-stop on engines they don't train on. Each net was trained from **3 random starts** (seeds 42, 1, 2) in every fold, and the deployable net is their average. Verdict rule as in the promotion gate: a net or blend replaces boosting only if it lowers capped CV RMSE by more than 0.5 cycles in at least 4 of 5 folds.

| Capped RMSE (cycles) | FD001 | FD002 | FD003 | FD004 |
|---|---|---|---|---|
| **HistGB, CV** | **10.21** | **11.54** | **10.62** | **12.13** |
| GRU (best net, 3-seed average), CV | 15.15 | 15.17 | 13.57 | 15.25 |
| TCN, CV | 15.90 | 16.12 | 15.93 | 16.76 |
| 1D-CNN, CV | 16.28 | 16.55 | 16.15 | 16.93 |
| HistGB + GRU average (best blend), CV | 11.45 | 12.34 | 10.89 | 12.60 |
| **HistGB, official test** | **10.23** | **10.68** | **10.85** | **10.96** |
| GRU, official test | 13.01 | 13.48 | 11.97 | 12.62 |
| HistGB + GRU, official test | 10.96 | 11.21 | 10.51 | 10.90 |

- **Verdict: boosting kept on all four subsets.** The best net is 3.0–4.9 cycles worse in CV and loses in every one of the 60 net-vs-HistGB fold comparisons; the CNN and TCN are 4.6–6.1 cycles worse. The best blend (HistGB + GRU) is still worse in CV on every subset (by 0.27–1.23 cycles).
- **Not a training failure:** no run reached the 60-epoch limit (best epoch 7–20, then early stopping); all 216 net fits took 21 minutes on a T4.
- **Seeds mattered, as expected:** single random starts scattered by up to 1.1 cycles of CV RMSE (CNN on FD001); averaging the 3 seeds improved each net by 0.2–0.7 cycles. A one-seed comparison would have been partly luck; here the gap is 3–6× larger than the seed scatter.
- **Why the trees win here:** a 30-cycle window can't see an engine's early-life baseline, while the trees get it as the *drift-from-own-baseline* feature. The ablation (9.6) prices exactly this: removing drift costs 3.3–5.5 cycles and using raw sensors only costs 6.4–8.9 — the same size as the nets' deficit. With 100–250 engines per subset, the nets also have little data to learn such features themselves.
- **One trap we avoided:** the HistGB + GRU blend happens to beat HistGB on the test set for FD003 and FD004. Adopting it for that reason would be choosing a model on the test set; the CV says it is worse, so it is not adopted.
- **Not tried:** longer windows or feeding the drift channel into the nets would likely narrow the gap, but would mostly rebuild the engineered features inside a heavier model. Given a 3–6 cycle deficit and a brief that asks for a lightweight system, we stopped here.

---

## 10. How the design evolved (evidence-driven decisions)

| Observation | Decision |
|---|---|
| First run: CRITICAL over-triggered (13/15 HIGH → CRITICAL) because a classifier rule was used for a distinction the classifier cannot make | CRITICAL now decided by the calibrated RUL lower bound only |
| First run: human review never triggered (0%) because a fixed width threshold (60) was never reached; intervals covered only 64% instead of 80% | Conformal calibration added; review threshold learned from CV (~10% of cases by design) |
| FD002 test RMSE 23 vs validation 13–14 | Error-by-band analysis: 81% from engines above the cap. Validation window widened to RUL 200; capped RMSE made a headline metric and the basis for selection/promotion |
| Widening the window weakened calibration (rows above the cap are covered automatically) and inflated the NASA score used by promotion | Calibration restricted to RUL ≤ 125; promotion uses NASA vs capped truth |
| Full refit on all machines tested | No measurable gain → kept off |
| External review: back-fill copied later cycles into earlier ones | Back-fill removed; a second causality test now covers cleaning on gap-injected data |
| External review: v1's model choice and calibration used engines from the "new" batch | Selection and calibration now run per version on that version's training engines only |
| External review: anomaly detector's "healthy" data was defined with hindsight (RUL > 125) | Retrained on each engine's early-life cycles (11–40) — no failure labels needed |
| External review: F2 threshold advertised but agent used 0.5 | Roles separated and documented: F2 → early-warning flag, 0.5 → agent HIGH rule |
| The early-life anomaly detector alerts earlier but raised false alarms from 5 to 14 (every extra one was "anomaly only", RUL model > 60) | Rule chosen on **validation** engines: an anomaly raises MEDIUM only if the RUL model confirms (≤ 100); otherwise a LOW watch item. Healthy-cycle orders fell 2–12×, near-failure detection unchanged; test false alarms back to 5 |
| 5-seed rerun: the superiority gate kept v1 in 3 of 20 runs, wrongly in 2 (v2 was better on test) | Gate switched to non-inferiority for same-design retrains; confirmed on Kaggle: v2 promoted in 12 of 12 runs |
| Ablation: HistGB + XGBoost average ~1% better | Too small to justify two models; not adopted |
| Deep sequence models (CNN, GRU, TCN, 3 seeds each) tested under the same folds and rule | 3–6 cycles worse in CV on every subset; boosting kept (section 9.9) |
| Rerun on a different Kaggle machine: GPU XGBoost differs in the last digits and flips the HistGB/XGBoost near-tie | Reproducibility claim narrowed to "decisions identical, capped RMSE within ±0.3"; CPU XGBoost + tie-break designed (11.2) |

---

## 11. Edge cases

### 11.1 Handled [Built]
- Missing values / gaps → per-machine fill + `imputed` flag → human review.
- Duplicate or out-of-order rows → de-duplicated and sorted.
- Non-numeric values → coerced and counted in the validation report.
- Constant / uninformative sensors → dropped automatically per subset.
- Outlier spikes → clipped to training quantiles.
- Machines with very short history → shrinking windows (no crash, no missing values) + human review below 15 cycles.
- Machines at the very start of life → drift baseline uses only cycles already seen.
- Rule conflicts → highest severity wins, all reasons logged; contradicting model signals → human review.
- Due date can't go negative (clamped to "now").
- Repeated alerts on the same machine → one open order, escalation supersedes.
- Retraining that makes things worse → promotion gate (incl. bootstrap check) keeps the old champion. Not yet observed rejecting in a run.
- Tracking backend unavailable → JSON fallback.
- One subset failing in the batch run → recorded, others continue.
- **Unseen operating condition** → mapped to the nearest known condition (not silent global statistics), flagged, human review.
- **Sensor fault vs machine fault** → dead / stuck / out-of-range sensors raise a separate `CHECK_SENSOR` ticket and send maintenance orders to review (section 4.7, demonstrated in 9.8).
- **Flip-flopping alerts** → hysteresis (5 calmer cycles before a downgrade); unhandled reviews flagged overdue after 10 cycles.

### 11.2 Known gaps and planned fixes [Designed]
1. **CRITICAL on suspect data is still dispatched.** In the fault demo, an unseen operating condition pushed one FD004 engine from MEDIUM to CRITICAL (RUL 51 → 10); by design CRITICAL dispatches anyway (flagged for review). Safety-first, but it can ground a healthy machine on bad data; a policy choice to revisit with real costs.
2. **Interval coverage is weaker at long RUL:** 72–75% in the 60–125 band on test (target 80%); 89–100% in the ≤ 30 band that CRITICAL relies on.
3. **Review timeout is unexercised in a meaningful way:** with no simulated reviewer, every review eventually goes overdue (6–38 per subset in the replay).
4. **Closed feedback loop** (section 12).
5. **Drift path never exercised.** Fix: inject synthetic drift (sensor offset, new regime) to demonstrate trigger, review and rollback.
6. **RMSE is not bit-reproducible across Kaggle machines** (section 9.7): GPU XGBoost differs in the last digits on different hosts, which can flip the HistGB/XGBoost near-tie (±0.3 capped RMSE; decisions unaffected). Fix: run XGBoost on CPU with a fixed thread count, and select XGBoost only if it beats HistGB by a clear CV margin.

---

## 12. The closed loop: feedback from maintenance [Designed]

1. Technician closes the work order with an outcome: **fault confirmed / no fault found / too late (failed before service)**.
2. The outcome is stored as a label with the prediction that triggered the order.
3. Periodic recalibration: false-alarm and miss rates per severity tier adjust the thresholds (within limits, logged as a new agent version).
4. Confirmed outcomes and maintenance events enter the training data. After maintenance, the machine's health history restarts (new baseline), so features and labels count from the last service, not from first commissioning.
5. **Censoring is handled explicitly:** once the system works, machines are serviced *before* failure, so true run-to-failure histories become rare. Retraining then uses survival-analysis-style targets (time to maintenance as a lower bound on life) together with confirmed-fault labels.
6. Retraining and promotion follow the same gate as section 7.2, now triggered also by feedback volume and by rising false-alarm or miss rates.

---

## 13. From simulation to a real fleet

**What transfers:** the pipeline structure, leakage-safe evaluation, per-regime normalisation (flight regimes → load / speed / route regimes), calibrated uncertainty with human review, the retraining and promotion loop, the rule-based agent and its audit trail.
**What is re-learned per fleet:** the models, sensor selection, RUL cap and windows, calibration margin, and agent thresholds (ideally set from real maintenance costs: cost of an unplanned failure vs an early service vs a human review).

Real-fleet differences the design accounts for:
- **Different sensors** (vibration, motor current, temperature, pressure, RPM at irregular rates) → resampling to a fixed step per asset type, then the same features.
- **No run-to-failure labels, maintenance resets** → anomaly detector works without labels from day one; RUL models start from maintenance records and improve through the feedback loop.
- **Real drift** (new machines, seasons, sensor replacements) → the PSI trigger and promotion gate are built for exactly this.
- **Cycles to calendar** → "due within N cycles" converted with each asset's usage rate and parts lead time.

**Deployment sketch [Designed], no cloud needed:** two levels, both on site.
- **On each machine (or its gateway):** features, scoring, the sensor-health check and the agent. The production model bundle (preprocessing, all models, thresholds) is **~10 MB per subset (measured: 9.4–10.0 MB)** and scores on CPU, so it runs offline; work orders queue until the link is back.
- **On a local server at the depot or plant:** pools that site's fleet data (a single machine sees too few failures to learn from), retrains, runs the promotion gate and pushes new bundles to the machines. A full retrain cycle took 4–6 minutes per subset on a Kaggle machine; on-site CPU timing is not yet measured. MLflow (local SQLite), the registry and the dashboard all run locally.
- Telemetry can reach the server over MQTT (or Kafka at larger sites); the agent writes work orders to the maintenance system (CMMS) through an API. A cloud is optional, for example to compare sites, not required.
**Safe rollout:** backtest on historical sensor + maintenance logs → shadow mode alongside the current process → limited rollout with human sign-off → widen automation as measured precision and recall hold.

---

## 14. Limitations (stated openly)
- C-MAPSS is simulated: clean one-row-per-cycle data, no maintenance resets, no real drift, at most two fault modes. The models will not transfer as-is; the design does.
- Headline numbers are from seed 42, and a 5-seed rerun confirms them (section 9.7); design choices are now backed by a cross-validated ablation (section 9.6), and the promotion gate uses a bootstrap check.
- Validation sets are small (20–52 machines), so promotion decisions are noisy.
- Agent thresholds (15 / 30 / 60 / 100 cycles) were set by judgement, then checked with a cost model (section 9.4): near cost-optimal and more robust than cost-tuned alternatives on this data, but not derived from real fleet costs, which are unknown.
- Fault mode per machine is unknown in the data, so recommended actions cannot say *which* module (HPC or fan) failed in FD003/FD004.
- The NASA score on FD002/FD004 is dominated by the healthy engines above the cap; it is reported, but decision metrics and capped RMSE describe the system better.
