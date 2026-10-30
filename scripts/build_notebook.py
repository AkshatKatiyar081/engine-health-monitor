"""Generate notebooks/pipeline_FD001.ipynb from the cell sources below.

Edit the cells here and re-run:  python scripts/build_notebook.py
Cells tagged "train" fit models; they are meant to run on Kaggle, not locally.
"""
import json
from pathlib import Path

CELLS = []
DASHBOARD_SRC = (Path(__file__).resolve().parent / "build_dashboard.py").read_text(encoding="utf-8")


def md(src):
    CELLS.append(("markdown", src.strip("\n"), []))


def code(src, *tags):
    CELLS.append(("code", src.strip("\n"), list(tags)))


md(r"""
# Predictive maintenance pipeline · NASA C-MAPSS FD001

End-to-end, closed-loop predictive-maintenance pipeline:

| Stage | What this notebook does |
|---|---|
| 1. Ingest | Finds the dataset, loads train / test / true-RUL files, validates the schema |
| 2. Clean + label | Types, duplicates, missing values (per-engine forward fill), RUL labels |
| 3. Features | Per-engine causal rolling features: mean, std, slope, EWMA, drift from the engine's own baseline |
| 4. Split | By **engine**, never by row: validation engines, an initial batch, and a "new data" batch |
| 5. Model selection | 5-fold GroupKFold CV comparing Ridge, HistGradientBoosting and XGBoost for RUL (chosen on capped RMSE) |
| 6. Retraining flow | v1 on the initial batch → new data arrives → drift check → v2 → promotion rule → champion |
| 6b. Production refit | the champion's design retrained on all training engines → `production` |
| 7. Evaluation | Official test metric (last cycle per engine): RMSE + NASA asymmetric score, plus classifier and anomaly metrics |
| 8. Triage agent | Rule-based (no LLM): severity, human-review exceptions, work orders |
| 9. Streaming replay | One engine fed cycle by cycle, showing when the agent escalates |
| 10. Export | Models, predictions, work orders, monitoring JSON, plots, MLflow store → one zip |

**Models**
- **RUL regressor**: predicts cycles left, trained on RUL capped at 125 (piecewise-linear target).
- **Quantile models (10% / 90%)**: give an interval, widened by conformal calibration on cross-validation folds so it really covers ~80% of engines. The agent acts on its lower bound and flags the most uncertain cases.
- **Early-warning classifier**: P(failure within 30 cycles), `class_weight="balanced"` for the ~15% minority class.
- **Anomaly detector**: Isolation Forest trained only on each engine's early-life cycles (11–40), so it needs no failure labels. Its false-alarm rate is evaluated on cycles with RUL > 125, a proxy for "healthy".

**Run on Kaggle**: attach the dataset (the `data/csv` folder or the raw `DataSet` .txt files), set *Accelerator → GPU T4 x2* and *Internet → On* (used once to install MLflow), then *Run all*. Change `SUBSET` in the config cell to run FD002–FD004.
""")

code(r"""
# MLflow is not preinstalled on Kaggle. If the install fails, the notebook falls back to JSON tracking.
import importlib.util, subprocess, sys

def ensure(module, pip_name=None):
    if importlib.util.find_spec(module) is None:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or module],
                           capture_output=True, text=True)
        print(f"pip install {pip_name or module}: {'ok' if r.returncode == 0 else 'FAILED'}")

ensure("mlflow")
""")

code(r"""
import json, shutil, subprocess, time, warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    import xgboost as xgb
except ImportError:
    xgb = None

SUBSET = "FD001"  # FD001 / FD002 / FD003 / FD004 (run_all_subsets.ipynb rewrites this line)
SEED = 42
RUN_TAG = ""  # run_all_subsets sets e.g. "_seed1" for extra seeds, so their outputs don't overwrite seed 42's

SUBSET_CONFIG = {
    "FD001": {"condition_norm": False, "fault_modes": 1},
    "FD002": {"condition_norm": True, "fault_modes": 1},
    "FD003": {"condition_norm": False, "fault_modes": 2},
    "FD004": {"condition_norm": True, "fault_modes": 2},
}

CFG = {
    "subset": SUBSET, **SUBSET_CONFIG[SUBSET],
    # labels
    "rul_cap": 125,             # piecewise-linear target: flat while the engine is healthy
    "fail_window": 30,          # "failing" class for the early-warning classifier: RUL <= 30
    # cleaning / features
    "min_unique": 2,            # drop sensors with <= 2 distinct values within a condition
    "clip_quantiles": (0.001, 0.999),
    # data-quality gate (all fitted on training engines)
    "op_margin_frac": 0.5,        # single-condition subsets: settings beyond the training range by this much = unseen
    "sensor_range_margin": 0.5,   # reading beyond the training min/max by 50% of the range = out of range
    "sensor_range_persist": 2,    # ... for at least this many consecutive cycles
    "sensor_stuck_cycles": 15,    # identical reading this long on a sensor that never flatlines in training = stuck
    "roll_window": 10, "slope_window": 20, "ewm_span": 10, "baseline_cycles": 10,
    # splits (by engine)
    "val_frac": 0.20, "new_frac": 0.30,
    "cv_folds": 5,
    # 'test-like' evaluation rows: mirror how the official test engines are cut.
    # Test RUL reaches ~195 in FD002/FD004, so the window covers RUL up to 200.
    "eval_min_cycles": 31, "eval_max_rul": 200,
    # anomaly detector: trained on early-life cycles (no failure labels needed; a real fleet would use the
    # first cycles after commissioning or maintenance). Starts after the rolling window is full.
    "anomaly_train_cycles": (11, 40), "anomaly_quantile": 0.99, "anomaly_persist": 3,
    "anomaly_healthy_rul": 125,  # evaluation only: false-alarm rate is measured on RUL > 125 (a proxy)
    # RUL interval: conformal calibration of the 10% / 90% quantile models
    "interval_coverage": 0.80, "review_width_quantile": 0.90,
    # retraining / promotion
    "retrain_min_new_frac": 0.20, "psi_alert": 0.20, "promote_max_nasa_ratio": 1.10,
    "promote_min_prob": 0.80,     # how sure the bootstrap must be (80% of resamples of the validation engines)
    # retraining here is always "same design, more data", so v2 is promoted unless it is clearly WORSE
    # (non-inferiority); "superiority" (v2 must be clearly better) is the stricter test for design changes
    "promotion_test": "non_inferiority",
    "noninferiority_margin": 0.5,  # cycles of capped RMSE: "clearly worse" means worse by more than this (< fold spread)
    # after promotion, retrain the winning design on every training engine and deploy that
    "full_refit": False,
}

ON_KAGGLE = Path("/kaggle/input").exists()
WORK = Path("/kaggle/working") if ON_KAGGLE else Path.cwd() / "outputs"
OUT = WORK / "artifacts" / f"{SUBSET}{RUN_TAG}"
(OUT / "plots").mkdir(parents=True, exist_ok=True)
(OUT / "models").mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)


def gpu_available():
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        return r.returncode == 0 and "GPU" in r.stdout
    except FileNotFoundError:
        return False


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (Path,)):
        return str(o)
    raise TypeError(type(o))


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=json_default))


GPU = gpu_available()
print(f"subset={SUBSET}  gpu={GPU}  xgboost={xgb.__version__ if xgb else 'not installed'}  outputs={OUT}")
""")

md(r"""
## 1. Ingest

Looks for `train_<SUBSET>.csv` (from `scripts/convert_to_csv.py`) or the raw `train_<SUBSET>.txt` under `/kaggle/input`, so either upload works. Raw files have no header: 26 space-separated columns (unit, cycle, 3 operating settings, 21 sensors). `RUL_<SUBSET>` holds one true RUL per test engine, in engine order.
""")

code(r"""
COLS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
OPS, SENSORS = COLS[2:5], COLS[5:]


def find_data_dir(subset):
    roots = [Path("/kaggle/input"), Path.cwd() / "data" / "csv", Path.cwd().parent / "data" / "csv",
             Path.cwd() / "DataSet", Path.cwd().parent / "DataSet"]
    for root in roots:
        if root.exists():
            for ext in ("csv", "txt"):
                hits = sorted(root.rglob(f"train_{subset}.{ext}"))
                if hits:
                    return hits[0].parent, ext
    raise FileNotFoundError(f"train_{subset}.csv / .txt not found under {[str(r) for r in roots]}")


def read_table(path, ext):
    if ext == "csv":
        return pd.read_csv(path)
    df = pd.read_csv(path, sep=r"\s+", header=None)
    return df.iloc[:, :len(COLS)].set_axis(COLS, axis=1)


def validate_raw(df, name):
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    num = df[COLS].apply(pd.to_numeric, errors="coerce")
    report = {
        "rows": len(df),
        "engines": int(num.unit.nunique()),
        "missing_cells": int(df[COLS].isna().sum().sum()),
        "non_numeric_cells": int(num.isna().sum().sum() - df[COLS].isna().sum().sum()),
        "duplicate_unit_cycle": int(num.duplicated(subset=["unit", "cycle"]).sum()),
        "cycle_gaps": int(num.sort_values(["unit", "cycle"]).groupby("unit").cycle.diff().dropna().ne(1).sum()),
    }
    if report["rows"] == 0:
        raise ValueError(f"{name}: empty file")
    return report


def load_subset(subset):
    d, ext = find_data_dir(subset)
    train = read_table(d / f"train_{subset}.{ext}", ext)
    test = read_table(d / f"test_{subset}.{ext}", ext)
    if ext == "csv":
        r = pd.read_csv(d / f"RUL_{subset}.csv")
        final = (r.sort_values("unit") if "unit" in r else r)["rul"].to_numpy()
    else:
        final = pd.read_csv(d / f"RUL_{subset}.txt", sep=r"\s+", header=None)[0].to_numpy()
    reports = {"train": validate_raw(train, "train"), "test": validate_raw(test, "test")}
    test_units = np.sort(pd.to_numeric(test.unit).unique())
    if len(final) != len(test_units):
        raise ValueError(f"RUL file has {len(final)} values but test has {len(test_units)} engines")
    rul_final = pd.DataFrame({"unit": test_units.astype(int), "rul_final": final.astype(int)})
    return train[COLS], test[COLS], rul_final, reports, d


raw_train, raw_test, rul_final, ingest_report, DATA_DIR = load_subset(SUBSET)
print("data dir:", DATA_DIR)
display(pd.DataFrame(ingest_report).T)
""")

md(r"""
## 2. Clean and label

- Coerce to numbers, drop duplicate `(unit, cycle)` rows, sort.
- Missing values: forward-fill **within each engine** (sensor readings are continuous in time). There is no back-fill, because that would copy later cycles into earlier ones. Gaps at the very start of an engine's history stay missing and become the training mean after normalisation. Every row with a missing value is flagged `imputed`, which the agent treats as a data-quality exception.
- Labels: train RUL = `last_cycle − cycle`; test RUL = `true_final_RUL + last_cycle − cycle`. `rul_capped = min(RUL, 125)`; `fail = RUL <= 30`.
""")

code(r"""
def basic_clean(df):
    df = df[COLS].apply(pd.to_numeric, errors="coerce").dropna(subset=["unit", "cycle"])
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    df = df.drop_duplicates(subset=["unit", "cycle"], keep="last").sort_values(["unit", "cycle"]).reset_index(drop=True)
    vals = OPS + SENSORS
    df["imputed"] = df[vals].isna().any(axis=1)
    # past-only: forward fill within each engine; leading gaps become the training mean after normalisation
    df[vals] = df.groupby("unit")[vals].ffill()
    return df


def add_labels(df, cfg, rul_final=None):
    df = df.copy()
    last = df.groupby("unit").cycle.transform("max")
    end = 0 if rul_final is None else df.unit.map(rul_final.set_index("unit").rul_final)
    df["rul"] = (end + last - df.cycle).astype(int)
    df["rul_capped"] = df.rul.clip(upper=cfg["rul_cap"])
    df["fail"] = (df.rul <= cfg["fail_window"]).astype(int)
    return df


train_df = add_labels(basic_clean(raw_train), CFG)
test_df = add_labels(basic_clean(raw_test), CFG, rul_final)

check = test_df.groupby("unit").rul.min().reset_index().merge(rul_final, on="unit")
assert (check.rul == check.rul_final).all(), "test labels do not match the RUL file"
print(f"train: {train_df.unit.nunique()} engines, {len(train_df):,} rows | test: {test_df.unit.nunique()} engines, {len(test_df):,} rows")
print(f"imputed rows: train {int(train_df.imputed.sum())}, test {int(test_df.imputed.sum())}")
print(f"class balance (fail = RUL <= {CFG['fail_window']}): {train_df.fail.mean():.1%} positive")
""")

md(r"""
## 3. Preprocessing and features

**Preprocessing** (fitted on training engines only, stored as plain dicts so it can be saved as JSON):
1. Operating condition = rounded `(op1, op2, op3)`. For FD001/FD003 there is one condition; for FD002/FD004 six.
2. Drop sensors with ≤ 2 distinct values within a condition (FD001: s1, s5, s6, s10, s16, s18, s19 → 14 sensors remain).
3. Z-score each sensor **within its condition** (FD001: a single global z-score).
4. Clip to the 0.1% / 99.9% training quantiles to tame noise spikes.

**Data-quality gate** (also fitted on training engines; flags only, never changes the data):
- **Unseen operating condition:** settings that match no training condition are mapped to the nearest known one (instead of silently using global statistics) and flagged; the agent sends such cases to human review.
- **Sensor health:** *dead* (no reading at all), *stuck* (identical for 15 cycles on a sensor that never flatlines in training — the check is learned per sensor, since some sensors are coarse by design), *out of range* (beyond the training min–max by half the range for 2+ cycles). On the real C-MAPSS data these flag nothing; section 10b injects faults to show them working.

**Features** (per engine, using only the current and past cycles, so they work on a live stream and on truncated test engines):
- raw z-value, rolling mean (10), rolling std (10), rolling slope (20), EWMA (span 10)
- `drift` = EWMA − the engine's own baseline (mean of its first 10 cycles). This removes each engine's initial wear offset.
- `cycle` (time in service)

Two causality checks below assert that no future data leaks into any row:
- features recomputed on truncated engines match the full-series features at the cut;
- the same holds for **cleaning plus features** on engines with injected gaps (including gaps at the start of the series), so the missing-value handling is past-only too.
""")

code(r"""
META = ["unit", "cycle", "condition", "imputed", "unseen_condition", "sensor_fault", "sensor_fault_detail",
        "rul", "rul_capped", "fail"]


def condition_key(df):
    return (df[OPS].round(0) + 0.0).astype(str).agg("/".join, axis=1)


def assign_condition(df, prep):
    # operating condition per row; settings never seen in training fall back to the nearest known condition and
    # are flagged, instead of silently using global statistics
    if not prep["condition_norm"]:
        lo = pd.Series({o: prep["op_range"][o][0] for o in OPS})
        hi = pd.Series({o: prep["op_range"][o][1] for o in OPS})
        margin = (hi - lo) * prep["quality"]["op_margin_frac"] + 0.01 * np.maximum(1.0, np.maximum(lo.abs(), hi.abs()))
        unseen = ((df[OPS] < lo - margin) | (df[OPS] > hi + margin)).any(axis=1)
        return pd.Series("all", index=df.index), unseen
    key = condition_key(df)
    unseen = ~key.isin(list(prep["stats"]))
    if unseen.any():
        names = list(prep["centers"])
        centers = np.array([prep["centers"][n] for n in names], dtype=float)
        scale = np.where(centers.std(axis=0) > 0, centers.std(axis=0), 1.0)
        x = df.loc[unseen, OPS].to_numpy(float)
        dist = np.nansum(((x[:, None, :] - centers[None]) / scale) ** 2, axis=2)
        nearest = np.array(names, dtype=object)[dist.argmin(axis=1)]
        nearest[np.isnan(x).all(axis=1)] = prep["default_condition"]
        key = key.copy()
        key.loc[unseen] = nearest
    return key, unseen


def _flat(frame, units, window):
    # identical for the whole window: max - min is exactly 0 (a rolling std leaves rounding residue up to ~1e-5)
    return (_rolling_by_unit(frame, units, window, "max") - _rolling_by_unit(frame, units, window, "min")) == 0


def _rolling_by_unit(frame, units, window, how):
    # per-engine trailing window statistic, aligned back to the frame's row order
    r = frame.groupby(units.to_numpy()).rolling(window, min_periods=window)
    return getattr(r, how)().reset_index(level=0, drop=True).reindex(frame.index)


def sensor_quality(df, prep):
    # past-only sensor-health checks: dead (never reported), stuck (flat for K cycles on a sensor that never
    # flatlines in training) and out of the physically seen range for >= persist cycles
    q, sensors = prep["quality"], prep["sensors"]
    raw = df[sensors].astype(float)
    lo = pd.DataFrame(prep["sensor_range"]["lo"]).T.loc[df["condition"], sensors].to_numpy()
    hi = pd.DataFrame(prep["sensor_range"]["hi"]).T.loc[df["condition"], sensors].to_numpy()
    span, x = hi - lo, raw.to_numpy()
    oor = (x < lo - q["range_margin"] * span) | (x > hi + q["range_margin"] * span)
    # an unseen operating condition shifts every reading; that is flagged separately, not blamed on the sensors
    oor &= ~df["unseen_condition"].to_numpy()[:, None]
    oor = pd.DataFrame(oor, index=df.index, columns=sensors).astype(float)
    oor = _rolling_by_unit(oor, df["unit"], q["range_persist"], "min").fillna(0).astype(bool)
    stuck = pd.DataFrame(False, index=df.index, columns=sensors)
    if prep["stuck_sensors"]:
        stuck[prep["stuck_sensors"]] = _flat(raw[prep["stuck_sensors"]], df["unit"], q["stuck_cycles"])
    dead = raw.isna()
    fault = (oor | stuck | dead).any(axis=1)
    detail = pd.Series("", index=df.index)
    for i in np.flatnonzero(fault.to_numpy()):
        row = df.index[i]
        parts = [f"{s} dead" for s in sensors if dead.at[row, s]] + \
                [f"{s} stuck" for s in sensors if stuck.at[row, s]] + \
                [f"{s} out of range" for s in sensors if oor.at[row, s]]
        detail.at[row] = "; ".join(parts)
    return fault, detail


def _normalise(df, prep, sensors):
    out = pd.DataFrame(index=df.index, columns=sensors, dtype=float)
    for c, g in df.groupby("condition"):
        st_ = prep["stats"].get(c, prep["global"])
        mu = pd.Series(st_["mean"])[sensors]
        sd = pd.Series(st_["std"])[sensors].replace(0, np.nan)
        out.loc[g.index] = ((g[sensors] - mu) / sd).to_numpy()
    return out.fillna(0.0)


def fit_prep(df, cfg):
    df = df.assign(condition=condition_key(df) if cfg["condition_norm"] else "all")
    stats = {c: {"mean": g[SENSORS].mean().to_dict(), "std": g[SENSORS].std().to_dict()}
             for c, g in df.groupby("condition")}
    glob = {"mean": df[SENSORS].mean().to_dict(), "std": df[SENSORS].std().to_dict()}
    nunique = df.groupby("condition")[SENSORS].nunique().median()
    sensors = [s for s in SENSORS if nunique[s] > cfg["min_unique"]]
    prep = {"sensors": sensors, "stats": stats, "global": glob, "condition_norm": cfg["condition_norm"]}
    z = _normalise(df, prep, sensors)
    lo, hi = cfg["clip_quantiles"]
    prep["clip"] = {s: (float(z[s].quantile(lo)), float(z[s].quantile(hi))) for s in sensors}
    prep["dropped"] = [s for s in SENSORS if s not in sensors]
    # what "normal" looks like, for the unseen-condition check and the sensor-health gate
    grouped = df.groupby("condition")
    prep["centers"] = {c: [float(v) for v in g[OPS].mean()] for c, g in grouped}
    prep["default_condition"] = str(df["condition"].value_counts().idxmax())
    prep["op_range"] = {o: [float(df[o].min()), float(df[o].max())] for o in OPS}
    prep["sensor_range"] = {"lo": grouped[sensors].min().T.to_dict(), "hi": grouped[sensors].max().T.to_dict()}
    k = cfg["sensor_stuck_cycles"]
    flat = _flat(df[sensors].astype(float), df["unit"], k)
    prep["stuck_sensors"] = [s for s in sensors if flat[s].mean() < 1e-4]  # never flat in training -> checkable
    prep["quality"] = {"op_margin_frac": cfg["op_margin_frac"], "range_margin": cfg["sensor_range_margin"],
                       "range_persist": cfg["sensor_range_persist"], "stuck_cycles": k}
    return prep


def apply_prep(df, prep):
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    condition, unseen = assign_condition(df, prep)
    df = df.assign(condition=condition, unseen_condition=unseen.to_numpy())
    fault, detail = sensor_quality(df, prep)
    df = df.assign(sensor_fault=fault.to_numpy(), sensor_fault_detail=detail.to_numpy())
    sensors = prep["sensors"]
    z = _normalise(df, prep, sensors)
    for s in sensors:
        z[s] = z[s].clip(*prep["clip"][s])
    return pd.concat([df[[c for c in META if c in df]], z], axis=1)


def _rolling_sum(a, w):
    c = np.cumsum(a, axis=0)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return out


def _rolling_slope(y, w):
    # least-squares slope over the last w cycles (fewer at the start of an engine's life)
    n_rows = len(y)
    t = np.arange(n_rows, dtype=float)[:, None]
    n = np.minimum(np.arange(1, n_rows + 1), w)[:, None].astype(float)
    st, stt = _rolling_sum(t, w), _rolling_sum(t * t, w)
    sy, sty = _rolling_sum(y, w), _rolling_sum(t * y, w)
    den = n * stt - st ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (n * sty - st * sy) / den
    return np.where(den > 1e-9, slope, 0.0)


def feature_cols(prep, cfg):
    w, sw, span = cfg["roll_window"], cfg["slope_window"], cfg["ewm_span"]
    sufs = ["", f"_mean{w}", f"_std{w}", f"_slope{sw}", f"_ewm{span}", "_drift"]
    return ["cycle"] + [f"{s}{suf}" for suf in sufs for s in prep["sensors"]]


def anomaly_cols(prep, cfg):
    return [f"{s}_mean{cfg['roll_window']}" for s in prep["sensors"]]


def build_features(dz, sensors, cfg):
    w, sw, span, base_n = cfg["roll_window"], cfg["slope_window"], cfg["ewm_span"], cfg["baseline_cycles"]
    blocks = []
    for _, g in dz.groupby("unit", sort=True):
        y = g[sensors].to_numpy(float)
        frame = pd.DataFrame(y)
        roll = frame.rolling(w, min_periods=1)
        ewm = frame.ewm(span=span, adjust=False).mean().to_numpy()
        base = np.cumsum(y, axis=0) / np.arange(1, len(y) + 1)[:, None]
        if len(y) > base_n:
            base[base_n:] = base[base_n - 1]
        blocks.append(np.hstack([y, roll.mean().to_numpy(), roll.std().fillna(0.0).to_numpy(),
                                 _rolling_slope(y, sw), ewm, ewm - base]))
    names = feature_cols({"sensors": sensors}, cfg)[1:]
    X = pd.DataFrame(np.vstack(blocks), columns=names, index=dz.index)
    return pd.concat([dz[[c for c in META if c in dz]], X], axis=1)


def featurize(df, prep, cfg):
    return build_features(apply_prep(df, prep), prep["sensors"], cfg)


def check_causality(df, prep, cfg, n_units=8, seed=SEED):
    rng = np.random.default_rng(seed)
    full = featurize(df, prep, cfg).set_index(["unit", "cycle"])
    feats = feature_cols(prep, cfg)[1:]
    for u in rng.choice(df.unit.unique(), n_units, replace=False):
        g = df[df.unit == u]
        cut = int(rng.integers(1, len(g) + 1))
        part = featurize(g.iloc[:cut], prep, cfg).iloc[-1]
        ref = full.loc[(u, int(part.cycle)), feats].to_numpy(float)
        assert np.allclose(part[feats].to_numpy(float), ref, atol=1e-8), f"feature leak at engine {u}"
    return True


def check_cleaning_causality(raw, prep, cfg, n_units=8, seed=SEED):
    # inject gaps (dense at the start of the series), then check cleaning + features stay past-only
    rng = np.random.default_rng(seed)
    feats = feature_cols(prep, cfg)[1:]
    for u in rng.choice(raw.unit.unique(), n_units, replace=False):
        g = raw[raw.unit == u].copy()
        gaps = rng.random((len(g), len(SENSORS))) < 0.05
        gaps[:3] |= rng.random((3, len(SENSORS))) < 0.5
        g[SENSORS] = g[SENSORS].mask(gaps)
        full = featurize(basic_clean(g), prep, cfg).set_index("cycle")
        cut = int(rng.integers(1, len(g) + 1))
        part = featurize(basic_clean(g.iloc[:cut]), prep, cfg).iloc[-1]
        ref = full.loc[int(part.cycle), feats].to_numpy(float)
        assert np.allclose(part[feats].to_numpy(float), ref, atol=1e-8), f"cleaning leak at engine {u}"
    return True


_prep_demo = fit_prep(train_df, CFG)
_feat_demo = featurize(train_df, _prep_demo, CFG)
print("sensors kept   :", _prep_demo["sensors"])
print("sensors dropped:", _prep_demo["dropped"])
print("conditions     :", len(_prep_demo["stats"]))
print(f"features       : {len(feature_cols(_prep_demo, CFG))} per row, matrix {_feat_demo.shape}")
assert not _feat_demo[feature_cols(_prep_demo, CFG)].isna().any().any()
print("causality check:", check_causality(train_df, _prep_demo, CFG))
print("cleaning causality check (injected gaps):", check_cleaning_causality(raw_train, _prep_demo, CFG))
""")

md(r"""
## 4. Split by engine

All splits are by engine id, so no engine contributes rows to both sides.
- **val** (20%): never trained on; used for model selection checks and the promotion decision.
- **initial** batch (50%): what we "had" at the start → model v1.
- **new** batch (30%): arrives later → triggers the retraining flow → model v2.

The official test set (`test_FD001` + `RUL_FD001`) is only used for reporting, never for choosing models.
""")

code(r"""
def split_units(df, cfg, seed=SEED):
    units = np.array(sorted(df.unit.unique()))
    np.random.default_rng(seed).shuffle(units)
    n_val, n_new = round(len(units) * cfg["val_frac"]), round(len(units) * cfg["new_frac"])
    return {"val": sorted(units[:n_val].tolist()),
            "new": sorted(units[n_val:n_val + n_new].tolist()),
            "initial": sorted(units[n_val + n_new:].tolist())}


SPLITS = split_units(train_df, CFG)
assert not (set(SPLITS["val"]) & set(SPLITS["new"]) | set(SPLITS["val"]) & set(SPLITS["initial"])
            | set(SPLITS["new"]) & set(SPLITS["initial"])), "engine leak between splits"
val_df = train_df[train_df.unit.isin(SPLITS["val"])].reset_index(drop=True)
print({k: len(v) for k, v in SPLITS.items()}, "engines")
""")

md(r"""
## 5. Experiment tracking

A thin wrapper: MLflow (SQLite backend → experiments, runs, model registry with a `champion` alias) when available, otherwise JSON files with the same calls, so the pipeline never stops on a tracking problem.
""")

code(r"""
class Tracker:
    def __init__(self, root, experiment):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = None
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            mlflow.set_tracking_uri(f"sqlite:///{(self.root / 'mlflow.db').as_posix()}")
            if mlflow.get_experiment_by_name(experiment) is None:
                mlflow.create_experiment(experiment, artifact_location=(self.root / "mlruns").as_uri())
            mlflow.set_experiment(experiment)
            self.mlflow, self.client = mlflow, MlflowClient()
            print("tracking: MLflow", mlflow.__version__, "->", self.root)
        except Exception as e:
            self.mlflow = None
            print("tracking: JSON fallback ->", self.root, f"({type(e).__name__}: {e})")

    def _load(self, name, default):
        p = self.root / name
        return json.loads(p.read_text()) if p.exists() else default

    def _save(self, name, obj):
        save_json(obj, self.root / name)

    @contextmanager
    def run(self, name, **tags):
        tags = {k: str(v) for k, v in tags.items()}
        if self.mlflow:
            with self.mlflow.start_run(run_name=name, tags=tags) as r:
                self.run_id = r.info.run_id
                yield self
        else:
            self.run_id = f"{name}-{int(time.time() * 1000)}"
            self._rec = {"run_id": self.run_id, "name": name, "tags": tags, "params": {}, "metrics": {},
                         "artifacts": [], "start": now_iso()}
            try:
                yield self
            finally:
                runs = self._load("runs.json", [])
                runs.append(self._rec)
                self._save("runs.json", runs)
        self.run_id = None

    def log_params(self, params):
        params = {k: str(v)[:500] for k, v in params.items()}
        if self.mlflow:
            self.mlflow.log_params(params)
        else:
            self._rec["params"].update(params)

    def log_metrics(self, metrics):
        metrics = {k: float(v) for k, v in metrics.items()
                   if isinstance(v, (int, float, np.number)) and np.isfinite(v)}
        if self.mlflow:
            self.mlflow.log_metrics(metrics)
        else:
            self._rec["metrics"].update(metrics)

    def log_artifact(self, path):
        if self.mlflow:
            self.mlflow.log_artifact(str(path))
        else:
            self._rec["artifacts"].append(str(path))

    def log_model(self, model, name):
        if self.mlflow:
            # XGBoost gets its native flavor; recent MLflow saves sklearn models with skops,
            # which rejects non-sklearn types, so sklearn models are pinned to cloudpickle
            if xgb is not None and isinstance(model, xgb.XGBModel):
                import mlflow.xgboost as flavor
                kwargs = {}
            else:
                import mlflow.sklearn as flavor
                kwargs = {"serialization_format": "cloudpickle"}
            try:
                info = flavor.log_model(model, name=name, **kwargs)
            except TypeError:
                info = flavor.log_model(model, artifact_path=name, **kwargs)
            return info.model_uri
        p = self.root / "models" / f"{self.run_id}_{name}.joblib"
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, p)
        self._rec["artifacts"].append(str(p))
        return str(p)

    def register(self, model_uri, name, **tags):
        if self.mlflow:
            mv = self.mlflow.register_model(model_uri, name)
            for k, v in tags.items():
                self.client.set_model_version_tag(name, mv.version, k, str(v))
            return int(mv.version)
        reg = self._load("registry.json", {})
        entry = reg.setdefault(name, {"versions": [], "aliases": {}})
        version = len(entry["versions"]) + 1
        entry["versions"].append({"version": version, "uri": model_uri, "created_at": now_iso(), "tags": tags})
        self._save("registry.json", reg)
        return version

    def set_alias(self, name, alias, version):
        if self.mlflow:
            self.client.set_registered_model_alias(name, alias, str(version))
        else:
            reg = self._load("registry.json", {})
            reg[name]["aliases"][alias] = version
            self._save("registry.json", reg)

    def get_alias(self, name, alias):
        try:
            if self.mlflow:
                return int(self.client.get_model_version_by_alias(name, alias).version)
            return self._load("registry.json", {}).get(name, {}).get("aliases", {}).get(alias)
        except Exception:
            return None


tracker = Tracker(WORK / f"tracking{RUN_TAG}", experiment=f"cmapss_{SUBSET}{RUN_TAG}")
REGISTRY_NAME = f"rul_{SUBSET.lower()}{RUN_TAG}"
""")

md(r"""
## 6. Models, training bundle and metrics

A **bundle** is everything needed to score new data: the fitted preprocessing dicts, feature list, RUL model, two quantile models, the early-warning classifier and the anomaly detector with its threshold.

**Metrics** (both RMSEs are headline numbers)
- `rmse`: against the true RUL, the standard C-MAPSS number.
- `rmse_capped`: against min(true RUL, 125). The model is trained on the capped target, so it never predicts above 125; for an engine with 190 cycles left, the ≥ 65-cycle error is unavoidable, and it is on the safe (early) side. FD002 / FD004 test sets have ~25% of engines above 125 (up to 195), against ~11–15% in FD001 / FD003, so their uncapped RMSE is dominated by healthy engines whose prediction changes no maintenance decision. **Model selection and promotion use `rmse_capped`**, the part of the error the model can actually reduce.
- `nasa_score`: the PHM08 asymmetric score, `exp(-d/13) − 1` for early and `exp(d/10) − 1` for late predictions (`d = predicted − true`). Lower is better; late predictions are punished harder because they mean unplanned failures. It is exponential, so the above-cap engines dominate it even more than RMSE; `nasa_capped_mean` scores against the capped truth and is what the promotion rule uses. `test_nasa_score` (uncapped, summed) stays the official reported number.
- `test_*`: the official evaluation, one prediction per test engine at its last observed cycle.
- `val_*`: validation engines, restricted to "test-like" rows (≥ 31 cycles observed, true RUL ≤ 200), which covers the full RUL range of all four test sets. Features are causal, so scoring a row equals scoring an engine truncated at that cycle.
""")

code(r"""
def make_rul_model(kind):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if kind == "hgb":
        return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                                             l2_regularization=1.0, early_stopping=False, random_state=SEED)
    if kind == "xgb":
        return xgb.XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=5, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, tree_method="hist",
                                device="cuda" if GPU else "cpu", random_state=SEED, n_jobs=-1)
    raise ValueError(kind)


def make_quantile(q):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=31, min_samples_leaf=20, early_stopping=False,
                                         random_state=SEED)


def make_classifier():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
                                          class_weight="balanced", early_stopping=False, random_state=SEED)


CANDIDATES = ["ridge", "hgb"] + (["xgb"] if xgb is not None else [])


def nasa_score(y_true, y_pred):
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)


def rul_metrics(y_true, y_pred, cap):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    d = y_pred - y_true
    s = nasa_score(y_true, y_pred)
    s_cap = nasa_score(np.minimum(y_true, cap), y_pred)
    return {"rmse": float(np.sqrt(np.mean(d ** 2))),
            "rmse_capped": float(np.sqrt(np.mean((y_pred - np.minimum(y_true, cap)) ** 2))),
            "mae": float(np.mean(np.abs(d))), "bias": float(np.mean(d)),
            "late_pct": float(np.mean(d > 0) * 100),
            "nasa_score": float(s.sum()), "nasa_mean": float(s.mean()),
            "nasa_capped_mean": float(s_cap.mean()), "n": int(len(d))}


def clf_metrics(y, p, threshold):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= threshold
    tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    two_classes = len(np.unique(y)) == 2
    return {"pr_auc": float(average_precision_score(y, p)) if two_classes else float("nan"),
            "roc_auc": float(roc_auc_score(y, p)) if two_classes else float("nan"),
            "precision": prec, "recall": rec,
            "f2": 5 * prec * rec / (4 * prec + rec) if prec + rec else 0.0,
            "positive_rate": float(y.mean())}


def anomaly_metrics(P, cfg):
    healthy = P.rul > cfg["anomaly_healthy_rul"]
    near = P.rul <= cfg["fail_window"]
    alarms = P[P.anomaly_persistent & ~healthy]
    first = alarms.groupby("unit").rul.max()
    return {"false_alarm_rate_healthy": float(P.loc[healthy, "anomaly_persistent"].mean()) if healthy.any() else float("nan"),
            "detection_rate_last30": float(P.loc[near, "anomaly_persistent"].mean()) if near.any() else float("nan"),
            "engines_detected_pct": float(len(first) / P.unit.nunique() * 100),
            "median_rul_at_first_alarm": float(first.median()) if len(first) else float("nan")}


def interval_coverage(P):
    # quantile models are trained on the capped target, so coverage is measured against it
    return float(((P.rul_capped >= P.rul_lo) & (P.rul_capped <= P.rul_hi)).mean())


def coverage_by_band(P, name, bands=((0, 30), (30, 60), (60, 125))):
    # coverage where decisions are made; CRITICAL relies on the lower bound when RUL <= 30
    out = {}
    for lo, hi in bands:
        b = P[(P.rul > lo) & (P.rul <= hi)]
        if len(b):
            out[f"{name}_coverage_rul_{lo}_{hi}"] = interval_coverage(b)
            out[f"{name}_below_lower_bound_rul_{lo}_{hi}"] = float((b.rul_capped < b.rul_lo).mean())
    return out


def bootstrap_gain(P_old, P_new, cfg, n_boot=2000, seed=SEED):
    # resample validation engines: how often is the new model better on capped RMSE, and how often is its
    # capped NASA score more than the allowed ratio worse?
    cap = cfg["rul_cap"]
    def per_engine(P):
        P = test_like(P, cfg)
        y = np.minimum(P.rul, cap)
        return P.assign(se=(P.rul_pred - y) ** 2, nasa=nasa_score(y, P.rul_pred)).groupby("unit")[["se", "nasa"]] \
                .agg(["sum", "count"])
    a = per_engine(P_old)
    b = per_engine(P_new).reindex(a.index)
    pick = np.random.default_rng(seed).integers(0, len(a), size=(n_boot, len(a)))
    mean = lambda s, col: s[(col, "sum")].to_numpy()[pick].sum(axis=1) / s[(col, "count")].to_numpy()[pick].sum(axis=1)
    diff = np.sqrt(mean(b, "se")) - np.sqrt(mean(a, "se"))
    nasa_ratio = mean(b, "nasa") / np.maximum(mean(a, "nasa"), 1e-12)
    return {"prob_new_better": float((diff < 0).mean()), "prob_new_worse": float((diff > 0).mean()),
            "prob_new_worse_by_margin": float((diff > cfg["noninferiority_margin"]).mean()),
            "rmse_diff_ci90": [float(np.quantile(diff, 0.05)), float(np.quantile(diff, 0.95))],
            "prob_nasa_clearly_worse": float((nasa_ratio > cfg["promote_max_nasa_ratio"]).mean()),
            "nasa_ratio_ci90": [float(np.quantile(nasa_ratio, 0.05)), float(np.quantile(nasa_ratio, 0.95))]}


def conformal_correction(y, lo, hi, coverage):
    # conformalized quantile regression: how far outside [lo, hi] the truth falls, at the target coverage
    scores = np.maximum(np.asarray(lo) - np.asarray(y), np.asarray(y) - np.asarray(hi))
    level = min(1.0, np.ceil((len(scores) + 1) * coverage) / len(scores))
    return float(np.quantile(scores, level))


def prefix(p, d):
    return {f"{p}_{k}": v for k, v in d.items()}


def latest(P):
    return P.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True)


def test_like(P, cfg):
    return P[(P.cycle >= cfg["eval_min_cycles"]) & (P.rul <= cfg["eval_max_rul"])]


def train_bundle(df, kind, cfg, clf_threshold, interval_correction=0.0):
    t0 = time.time()
    prep = fit_prep(df, cfg)
    F = featurize(df, prep, cfg)
    feats, acols = feature_cols(prep, cfg), anomaly_cols(prep, cfg)
    X, y = F[feats], F.rul_capped
    first, last = cfg["anomaly_train_cycles"]
    healthy = F.cycle.between(first, last)  # early life: no failure labels used
    iso = IsolationForest(n_estimators=300, random_state=SEED, n_jobs=-1).fit(F.loc[healthy, acols])
    return {
        "kind": kind, "subset": cfg["subset"], "cfg": dict(cfg), "prep": prep,
        "features": feats, "anomaly_features": acols,
        "rul_model": make_rul_model(kind).fit(X, y),
        "q_lo": make_quantile(0.1).fit(X, y), "q_hi": make_quantile(0.9).fit(X, y),
        "interval_correction": float(interval_correction),
        "classifier": make_classifier().fit(X, F.fail), "clf_threshold": float(clf_threshold),
        "iso": iso,
        "anomaly_threshold": float(np.quantile(-iso.score_samples(F.loc[healthy, acols]), cfg["anomaly_quantile"])),
        "units": sorted(int(u) for u in df.unit.unique()), "n_rows": int(len(df)),
        "train_seconds": round(time.time() - t0, 1), "created_at": now_iso(),
    }


def predict_bundle(b, df):
    cfg, cap = b["cfg"], b["cfg"]["rul_cap"]
    F = featurize(df, b["prep"], cfg)
    X = F[b["features"]]
    out = F[[c for c in META if c in F]].copy()
    pred = np.clip(b["rul_model"].predict(X), 0, cap)
    q = b.get("interval_correction", 0.0)
    out["rul_pred"] = pred
    out["rul_lo"] = np.minimum(np.clip(b["q_lo"].predict(X) - q, 0, cap), pred)
    out["rul_hi"] = np.maximum(np.clip(b["q_hi"].predict(X) + q, 0, cap), pred)
    out["p_fail"] = b["classifier"].predict_proba(X)[:, 1]
    out["fail_alert"] = out.p_fail >= b["clf_threshold"]
    out["anomaly_score"] = -b["iso"].score_samples(F[b["anomaly_features"]])
    out["anomaly"] = out.anomaly_score > b["anomaly_threshold"]
    k, w = cfg["anomaly_persist"], cfg["roll_window"]
    g = out.groupby("unit")
    out["anomaly_persistent"] = g.anomaly.transform(lambda a: a.astype(int).rolling(k, min_periods=k).min().fillna(0)).astype(bool)
    out["recent_imputed"] = g.imputed.transform(lambda a: a.astype(int).rolling(w, min_periods=1).max()).astype(bool)
    out["recent_unseen"] = g.unseen_condition.transform(lambda a: a.astype(int).rolling(w, min_periods=1).max()).astype(bool)
    out["cycles_observed"] = g.cumcount() + 1
    return out


def evaluate_bundle(b, val_df, test_df):
    # val_df=None skips validation metrics, for a model that has trained on the validation engines
    cfg, cap = b["cfg"], b["cfg"]["rul_cap"]
    Pt = predict_bundle(b, test_df)
    last = latest(Pt)
    m = prefix("test", rul_metrics(last.rul, last.rul_pred, cap))
    m["test_interval_coverage"] = interval_coverage(last)
    m.update(coverage_by_band(last, "test"))
    Pv = None
    if val_df is not None:
        Pv = predict_bundle(b, val_df)
        tl = test_like(Pv, cfg)
        m.update(prefix("val", rul_metrics(tl.rul, tl.rul_pred, cap)))
        m.update(prefix("val_all", rul_metrics(Pv.rul, Pv.rul_pred, cap)))
        m["val_interval_coverage"] = interval_coverage(tl)
        m.update(coverage_by_band(tl, "val"))
        m.update(prefix("val_clf", clf_metrics(Pv.fail, Pv.p_fail, b["clf_threshold"])))
        m.update(prefix("val_anom", anomaly_metrics(Pv, cfg)))
    return m, Pv, Pt


def show_metrics(m, keys=("val_rmse_capped", "val_rmse", "val_nasa_capped_mean", "test_rmse_capped", "test_rmse",
                          "test_nasa_score", "test_mae", "test_bias",
                          "test_late_pct", "val_interval_coverage", "test_interval_coverage", "val_clf_pr_auc", "val_clf_recall",
                          "val_clf_precision", "val_anom_false_alarm_rate_healthy", "val_anom_median_rul_at_first_alarm")):
    return pd.Series({k: round(m[k], 3) for k in keys if k in m})
""")

md(r"""
## 7. Model selection and calibration, per version (5-fold GroupKFold)

Selection and calibration run **once per model version, on that version's training engines only**: v1 uses the initial batch alone, and v2 repeats the whole step on initial + new. No version's choices use data that "arrived" after it, and each retrain recalibrates the full bundle (model type, thresholds, interval margin, review threshold) together.

Preprocessing is re-fitted inside every fold, so fold statistics never see the held-out engines. The fold predictions (from engines each model never saw) are also used to set three operating values:
- **Classifier alert threshold**: the one that maximises **F2** (recall weighted 2× over precision, because a missed failure costs more than an extra inspection).
- **Interval correction** (conformalized quantile regression): the 10% / 90% quantile models are too confident out of sample, so their interval is widened by the margin that makes it cover 80% of held-out test-like rows with true RUL ≤ 125 (above the cap the truth is covered almost automatically and would dilute the margin).
- **Review threshold**: the 90th percentile of calibrated interval width among rows likely to get a work order (predicted RUL ≤ 60). The 10% most uncertain cases go to a human.
""")

code(r"""
def select_and_calibrate(units, label):
    # CV over the given engines only: picks the RUL model and sets the classifier, interval and review values
    dev = train_df[train_df.unit.isin(units)].reset_index(drop=True)
    cap = CFG["rul_cap"]
    fold_frames = []
    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=CFG["cv_folds"]).split(dev, groups=dev.unit)):
        tr, va = dev.iloc[tr_idx], dev.iloc[va_idx]
        prep = fit_prep(tr, CFG)
        Ftr, Fva = featurize(tr, prep, CFG), featurize(va, prep, CFG)
        feats = feature_cols(prep, CFG)
        o = Fva[["unit", "cycle", "rul", "rul_capped", "fail"]].copy()
        o["fold"] = fold
        for kind in CANDIDATES:
            o[kind] = np.clip(make_rul_model(kind).fit(Ftr[feats], Ftr.rul_capped).predict(Fva[feats]), 0, cap)
        o["p_fail"] = make_classifier().fit(Ftr[feats], Ftr.fail).predict_proba(Fva[feats])[:, 1]
        o["q_lo"] = make_quantile(0.1).fit(Ftr[feats], Ftr.rul_capped).predict(Fva[feats])
        o["q_hi"] = make_quantile(0.9).fit(Ftr[feats], Ftr.rul_capped).predict(Fva[feats])
        fold_frames.append(o)
        print(f"[{label}] fold {fold}: {va.unit.nunique()} engines held out  ({time.time() - t0:.0f}s)")
    oof = pd.concat(fold_frames, ignore_index=True)

    oof_tl = test_like(oof, CFG)
    cv_results = pd.DataFrame({k: rul_metrics(oof_tl.rul, oof_tl[k], cap) for k in CANDIDATES}).T
    kind = cv_results.rmse_capped.idxmin()

    prec, rec, thr = precision_recall_curve(oof.fail, oof.p_fail)
    f2 = 5 * prec * rec / np.maximum(4 * prec + rec, 1e-9)
    clf_threshold = float(thr[np.argmax(f2[:-1])])
    cv_clf = clf_metrics(oof.fail, oof.p_fail, clf_threshold)

    # Calibrate only on rows with true RUL <= cap: above it the capped truth (125) sits at the clipped upper
    # bound and is covered almost automatically, which would shrink the margin where it matters.
    oof_cal = oof_tl[oof_tl.rul <= cap]
    raw_cover = float(((oof_cal.rul_capped >= oof_cal.q_lo) & (oof_cal.rul_capped <= oof_cal.q_hi)).mean())
    interval_q = conformal_correction(oof_cal.rul_capped, oof_cal.q_lo, oof_cal.q_hi, CFG["interval_coverage"])
    lo_cal = np.minimum(np.clip(oof.q_lo - interval_q, 0, cap), oof[kind])
    hi_cal = np.maximum(np.clip(oof.q_hi + interval_q, 0, cap), oof[kind])
    cal_cover = float(((oof.rul_capped >= lo_cal) & (oof.rul_capped <= hi_cal)).loc[oof_cal.index].mean())
    likely_orders = oof[kind] <= 60
    review_width = float(np.quantile((hi_cal - lo_cal)[likely_orders], CFG["review_width_quantile"]))
    cv_interval = {"raw_coverage": raw_cover, "calibrated_coverage": cal_cover,
                   "correction_cycles": interval_q, "review_width": review_width}

    for k in CANDIDATES:
        with tracker.run(f"{SUBSET}_{label}_cv_{k}", subset=SUBSET, stage=f"{label}_model_selection", model=k):
            tracker.log_params({"model": k, "folds": CFG["cv_folds"], "engines": len(units), **CFG})
            tracker.log_metrics(prefix("cv", cv_results.loc[k].to_dict()))
    with tracker.run(f"{SUBSET}_{label}_cv_calibration", subset=SUBSET, stage=f"{label}_model_selection"):
        tracker.log_params({"clf_threshold": clf_threshold, "class_weight": "balanced", "fail_window": CFG["fail_window"],
                            "target_coverage": CFG["interval_coverage"], "review_width_quantile": CFG["review_width_quantile"]})
        tracker.log_metrics({**prefix("cv_clf", cv_clf), **prefix("cv_interval", cv_interval)})

    display(cv_results.round(3))
    print(f"[{label}] {len(units)} engines | best RUL model: {kind} | classifier threshold (max F2): {clf_threshold:.3f} | "
          f"CV PR-AUC {cv_clf['pr_auc']:.3f}, recall {cv_clf['recall']:.2f}, precision {cv_clf['precision']:.2f}")
    print(f"[{label}] interval: raw coverage {raw_cover:.2f} -> widened by {interval_q:.1f} cycles each side -> "
          f"{cal_cover:.2f} (target {CFG['interval_coverage']:.2f}) | review if width > {review_width:.1f} cycles")
    return {"label": label, "engines": len(units), "kind": kind, "clf_threshold": clf_threshold,
            "interval_q": interval_q, "review_width": review_width,
            "cv_results": cv_results, "cv_clf": cv_clf, "cv_interval": cv_interval}


SEL_V1 = select_and_calibrate(SPLITS["initial"], "v1")
""", "train")

md(r"""
## 8. Retraining flow: v1 → new data → v2 → promotion

1. **v1** is trained on the initial batch with the choices from section 7, registered, and set as `champion`.
2. **New data arrives** (the "new" batch). It is validated and compared with the reference data using **PSI** per sensor (drift).
3. **Retrain trigger**: new engines ≥ 20% of the current training set, or any sensor PSI > 0.2.
4. **v2**: selection and calibration are re-run on initial + new, then v2 is trained and registered. The model type and thresholds may differ from v1's.
5. **Promotion rule (non-inferiority)**: v2 is the same design trained on more data, so it becomes `champion` **unless it is clearly worse**: v1 stays only if v2's validation capped RMSE is higher **by more than 0.5 cycles** (below the 0.6–1.2-cycle fold-to-fold spread) in at least 80% of 2,000 bootstrap resamples of the validation engines, or its validation NASA score (against the capped truth) is more than 10% worse in at least 80% of them. A stricter *superiority* test (v2 must be clearly better) is available via `CFG["promotion_test"]` for genuine design changes. The test set is not used for this decision. (A 5-seed experiment showed the superiority test rejected two v2 models that were in fact better, so it suits design changes, not "more data" retrains.)
6. **Rollback**: `production` is a registry alias, so rolling back means pointing it at the previous version; the old bundle (model and thresholds) comes back as a unit.
""")

code(r"""
history = []


def train_and_log(units, label, sel, validate=True, **tags):
    # sel: the selection/calibration result this version is built with (section 7)
    data = train_df[train_df.unit.isin(units)].reset_index(drop=True)
    b = train_bundle(data, sel["kind"], CFG, sel["clf_threshold"], sel["interval_q"])
    b.update({"label": label, "selection": sel, "review_width": sel["review_width"]})
    m, Pv, Pt = evaluate_bundle(b, val_df if validate else None, test_df)
    path = OUT / "models" / f"bundle_{label}.joblib"
    with tracker.run(f"{SUBSET}_{label}", subset=SUBSET, stage=label, **tags):
        tracker.log_params({"model": sel["kind"], "train_engines": len(units), "train_rows": len(data),
                            "selected_on": f"{sel['label']} ({sel['engines']} engines)",
                            "clf_threshold": sel["clf_threshold"], "interval_correction": sel["interval_q"],
                            "review_width": sel["review_width"], **CFG})
        tracker.log_metrics(m)
        uri = tracker.log_model(b["rul_model"], "rul_model")
        version = tracker.register(uri, REGISTRY_NAME, bundle=path.name, label=label)
        b["registry_version"] = version
        joblib.dump(b, path)
        tracker.log_artifact(path)
    history.append({"version": version, "label": label, "model": sel["kind"], "trained_at": b["created_at"],
                    "train_engines": len(units), "train_seconds": b["train_seconds"], **m})
    print(f"{label}: registered {REGISTRY_NAME} v{version}  ({b['train_seconds']}s)")
    return b, m, Pv, Pt


b1, m1, Pv1, Pt1 = train_and_log(SPLITS["initial"], "v1_initial", SEL_V1, trigger="initial training")
tracker.set_alias(REGISTRY_NAME, "champion", b1["registry_version"])
champion = {"bundle": b1, "metrics": m1, "Pv": Pv1, "Pt": Pt1}
display(show_metrics(m1).to_frame("v1"))
""", "train")

code(r"""
def psi(ref, new, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    r = np.histogram(np.clip(ref, edges[0], edges[-1]), edges)[0] / len(ref)
    n = np.histogram(np.clip(new, edges[0], edges[-1]), edges)[0] / len(new)
    r, n = np.clip(r, 1e-4, None), np.clip(n, 1e-4, None)
    return float(np.sum((n - r) * np.log(n / r)))


# new data arrives: validate it, then compare it with the data v1 was trained on
raw_new = raw_train[raw_train.unit.isin(SPLITS["new"])]
new_report = validate_raw(raw_new, "new batch")
ref_z = apply_prep(train_df[train_df.unit.isin(SPLITS["initial"])], b1["prep"])
new_z = apply_prep(train_df[train_df.unit.isin(SPLITS["new"])], b1["prep"])
drift = pd.Series({s: psi(ref_z[s], new_z[s]) for s in b1["prep"]["sensors"]}).sort_values(ascending=False)

new_frac = len(SPLITS["new"]) / len(SPLITS["initial"])
trigger_reasons = []
if new_frac >= CFG["retrain_min_new_frac"]:
    trigger_reasons.append(f"{len(SPLITS['new'])} new engines = {new_frac:.0%} of training set (>= {CFG['retrain_min_new_frac']:.0%})")
if drift.max() > CFG["psi_alert"]:
    trigger_reasons.append(f"drift: PSI {drift.max():.2f} on {drift.idxmax()} (> {CFG['psi_alert']})")
print("new batch:", new_report)
print(f"max PSI {drift.max():.3f} ({drift.idxmax()}) | retrain triggered: {bool(trigger_reasons)}")
for r in trigger_reasons:
    print("  -", r)

promotion = {"triggered": bool(trigger_reasons), "trigger_reasons": trigger_reasons, "promoted": False}
if trigger_reasons:
    SEL_V2 = select_and_calibrate(SPLITS["initial"] + SPLITS["new"], "v2")
    b2, m2, Pv2, Pt2 = train_and_log(SPLITS["initial"] + SPLITS["new"], "v2_retrained", SEL_V2,
                                     trigger="; ".join(trigger_reasons))
    better_rmse = m2["val_rmse_capped"] < m1["val_rmse_capped"]
    boot = bootstrap_gain(Pv1, Pv2, CFG)
    sure = CFG["promote_min_prob"]
    if CFG["promotion_test"] == "non_inferiority":
        # same design, more data: promote unless v2 is clearly worse — RMSE worse by more than the margin, or NASA
        # more than the allowed ratio worse — in at least `sure` of the resamples (ties and tiny differences = not worse)
        robust = boot["prob_new_worse_by_margin"] < sure
        nasa_ok = boot["prob_nasa_clearly_worse"] < sure
        promote = robust and nasa_ok
    else:
        # superiority (design changes): v2 must be clearly better, and its NASA score within the limit
        robust = boot["prob_new_better"] >= sure
        nasa_ok = m2["val_nasa_capped_mean"] <= m1["val_nasa_capped_mean"] * CFG["promote_max_nasa_ratio"]
        promote = better_rmse and robust and nasa_ok
    promotion.update({"challenger_version": b2["registry_version"], "champion_version": b1["registry_version"],
                      "val_rmse_capped": [m1["val_rmse_capped"], m2["val_rmse_capped"]],
                      "val_rmse": [m1["val_rmse"], m2["val_rmse"]],
                      "val_nasa_capped_mean": [m1["val_nasa_capped_mean"], m2["val_nasa_capped_mean"]],
                      "val_nasa_mean": [m1["val_nasa_mean"], m2["val_nasa_mean"]],
                      "better_rmse": better_rmse, "nasa_within_limit": nasa_ok,
                      "promotion_test": CFG["promotion_test"],
                      "bootstrap_prob_new_better": boot["prob_new_better"], "bootstrap_prob_new_worse": boot["prob_new_worse"],
                      "bootstrap_prob_new_worse_by_margin": boot["prob_new_worse_by_margin"],
                      "bootstrap_rmse_diff_ci90": boot["rmse_diff_ci90"], "robust_gain": bool(robust),
                      "bootstrap_prob_nasa_clearly_worse": boot["prob_nasa_clearly_worse"],
                      "bootstrap_nasa_ratio_ci90": boot["nasa_ratio_ci90"],
                      "promoted": bool(promote), "decided_at": now_iso()})
    print(f"[{CFG['promotion_test']}] v2 better in {boot['prob_new_better']:.0%} of bootstrap resamples "
          f"(RMSE change, 90% CI {boot['rmse_diff_ci90'][0]:+.2f} to {boot['rmse_diff_ci90'][1]:+.2f} cycles); "
          f"NASA more than {CFG['promote_max_nasa_ratio']:.2f}x worse in {boot['prob_nasa_clearly_worse']:.0%}")
    if promotion["promoted"]:
        tracker.set_alias(REGISTRY_NAME, "champion", b2["registry_version"])
        champion = {"bundle": b2, "metrics": m2, "Pv": Pv2, "Pt": Pt2}
    display(pd.DataFrame({"v1": show_metrics(m1), "v2": show_metrics(m2)}))

with tracker.run(f"{SUBSET}_promotion", subset=SUBSET, stage="promotion"):
    tracker.log_params({k: v for k, v in promotion.items() if not isinstance(v, list)})
    tracker.log_metrics(prefix("psi", drift.to_dict()))

CHAMPION_VERSION = champion["bundle"]["registry_version"]
print(f"promoted: {promotion['promoted']} -> champion = v{CHAMPION_VERSION} "
      f"(registry alias says v{tracker.get_alias(REGISTRY_NAME, 'champion')})")
""", "train")

md(r"""
## 8b. Production refit on all training engines

The champion was validated on 80 engines while 20 were held out. Once the promotion decision is made, the **same design** (model type, features, thresholds, interval correction) is retrained on all training engines and registered with the `production` alias. This is a policy fixed in advance (`CFG["full_refit"]`), not a choice made by comparing test scores, which would tune the model to the test set.

The refit has seen the validation engines, so it gets no validation metrics. Its test score is reported next to the champion's for transparency. Validation charts and the streaming replay (sections 9 and 11) keep using the champion, which never saw those engines.
""")

code(r"""
if CFG["full_refit"]:
    all_units = sorted(int(u) for u in train_df.unit.unique())
    b3, m3, _, Pt3 = train_and_log(all_units, "v3_full_refit", champion["bundle"]["selection"], validate=False,
                                   trigger=f"refit of champion v{CHAMPION_VERSION} design on all {len(all_units)} engines")
    production = {"bundle": b3, "metrics": m3, "Pt": Pt3}
else:
    production = {k: champion[k] for k in ("bundle", "metrics", "Pt")}
PRODUCTION_VERSION = production["bundle"]["registry_version"]
tracker.set_alias(REGISTRY_NAME, "production", PRODUCTION_VERSION)

compare_keys = ["test_rmse_capped", "test_rmse", "test_mae", "test_bias", "test_late_pct", "test_nasa_score", "test_interval_coverage"]
refit_comparison = pd.DataFrame({
    f"champion v{CHAMPION_VERSION} ({len(champion['bundle']['units'])} engines)": show_metrics(champion["metrics"], compare_keys),
    f"production v{PRODUCTION_VERSION} ({len(production['bundle']['units'])} engines)": show_metrics(production["metrics"], compare_keys),
})
display(refit_comparison)
print(f"aliases: champion = v{tracker.get_alias(REGISTRY_NAME, 'champion')}, "
      f"production = v{tracker.get_alias(REGISTRY_NAME, 'production')}")
""", "train")

md(r"""
## 9. Evaluation

The official numbers are `test_rmse_capped`, `test_rmse` and `test_nasa_score` of the **production** model: one prediction per test engine at its last observed cycle, against the subset's `RUL_` file. The error table by true-RUL band shows where the uncapped error comes from. Validation-based charts (engine trajectories, classifier precision/recall) use the **champion**, because the production refit has trained on those engines.
""")

code(r"""
b_c, m_c, Pt_c = production["bundle"], production["metrics"], production["Pt"]
Pv_c, m_v = champion["Pv"], champion["metrics"]
fleet = latest(Pt_c)
display(pd.DataFrame({f"production v{PRODUCTION_VERSION} (test)": show_metrics(m_c),
                      f"champion v{CHAMPION_VERSION} (validation)": show_metrics(m_v)}).reindex(show_metrics(m_v).index))


def error_by_band(P, cap, edges=(0, 30, 60, 100, 125, 150, 200)):
    # where the test error comes from; bands above the cap carry the unavoidable, early-side error
    d = P.assign(err=P.rul_pred - P.rul, band=pd.cut(P.rul, list(edges)).astype(str))
    t = d.groupby("band", sort=False).agg(engines=("err", "size"), rmse=("err", lambda e: np.sqrt((e ** 2).mean())),
                                          bias=("err", "mean"))
    t["share_of_sq_error_pct"] = d.groupby("band", sort=False).err.apply(lambda e: (e ** 2).sum()) / (d.err ** 2).sum() * 100
    return t.loc[[b for b in d.sort_values("rul").band.unique()]].round(1)


band_table = error_by_band(fleet, CFG["rul_cap"])
print(f"test engines with true RUL > {CFG['rul_cap']}: {int((fleet.rul > CFG['rul_cap']).sum())} of {len(fleet)}")
display(band_table)


def savefig(fig, name):
    p = OUT / "plots" / f"{name}.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.show()
    return p


plots = []

# 1. every test engine: true vs predicted RUL with the 10-90% interval
f = fleet.sort_values("rul").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(f.index, f.rul_lo, f.rul_hi, alpha=0.25, label="10-90% interval")
ax.plot(f.index, f.rul, "k-", lw=1.5, label="true RUL")
ax.plot(f.index, f.rul_pred, "o", ms=3, label="predicted RUL")
ax.set(xlabel="test engines (sorted by true RUL)", ylabel="RUL (cycles)",
       title=f"{SUBSET} test: RMSE {m_c['test_rmse_capped']:.1f} capped / {m_c['test_rmse']:.1f} true, "
             f"NASA score {m_c['test_nasa_score']:.0f}")
ax.axhline(CFG["rul_cap"], color="grey", ls=":", lw=1, label=f"RUL cap {CFG['rul_cap']}")
ax.legend()
plots.append(savefig(fig, "test_true_vs_pred"))

# 2. error distribution: positive = late (dangerous)
err = fleet.rul_pred - fleet.rul
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].scatter(fleet.rul, fleet.rul_pred, s=12)
lim = [0, max(fleet.rul.max(), fleet.rul_pred.max()) + 5]
axes[0].plot(lim, lim, "k--", lw=1)
axes[0].set(xlabel="true RUL", ylabel="predicted RUL", title="predicted vs true")
axes[1].hist(err[err <= 0], bins=20, alpha=0.7, label="early (safe side)")
axes[1].hist(err[err > 0], bins=20, alpha=0.7, label="late (risk)")
axes[1].set(xlabel="predicted − true (cycles)", ylabel="engines", title="error distribution")
axes[1].legend()
plots.append(savefig(fig, "test_errors"))

# 3. validation engines over their whole life
show_units = sorted(Pv_c.unit.unique())[:4]
fig, axes = plt.subplots(1, len(show_units), figsize=(16, 3.5), sharey=True)
for ax, u in zip(np.atleast_1d(axes), show_units):
    e = Pv_c[Pv_c.unit == u]
    ax.fill_between(e.cycle, e.rul_lo, e.rul_hi, alpha=0.25)
    ax.plot(e.cycle, e.rul_capped, "k-", lw=1, label="true (capped)")
    ax.plot(e.cycle, e.rul_pred, lw=1.5, label="predicted")
    a = e[e.anomaly_persistent]
    ax.plot(a.cycle, np.zeros(len(a)) + 2, "|", color="red", ms=8, label="anomaly")
    ax.set(title=f"val engine {u} · champion v{CHAMPION_VERSION}", xlabel="cycle")
np.atleast_1d(axes)[0].set_ylabel("RUL")
np.atleast_1d(axes)[0].legend(fontsize=8)
plots.append(savefig(fig, "val_trajectories"))

# 4. early-warning classifier precision/recall on validation engines
p_, r_, _ = precision_recall_curve(Pv_c.fail, Pv_c.p_fail)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(r_, p_)
ax.scatter([m_v["val_clf_recall"]], [m_v["val_clf_precision"]], color="red", zorder=3, label="chosen threshold")
ax.set(xlabel="recall", ylabel="precision", title=f"P(fail ≤ {CFG['fail_window']}) · PR-AUC {m_v['val_clf_pr_auc']:.3f}")
ax.legend()
plots.append(savefig(fig, "classifier_pr"))

with tracker.run(f"{SUBSET}_evaluation", subset=SUBSET, stage="evaluation", version=PRODUCTION_VERSION):
    tracker.log_metrics(m_c)
    for p in plots:
        tracker.log_artifact(p)
""", "train")

md(r"""
## 10. Triage agent (rule-based, no LLM)

Reads the production model's output for each engine and decides what to do. Rules are evaluated top-down; the highest severity that fires wins, and every rule that fired is recorded as a reason.

| Severity | Fires when (any) | Action | Work order |
|---|---|---|---|
| CRITICAL | calibrated lower-bound RUL ≤ 15 | remove from service, overhaul now | P1, due now |
| HIGH | predicted RUL ≤ 30, or P(fail ≤ 30) ≥ 0.50 | schedule maintenance, reserve parts | P2, due before `lower bound − 10` |
| MEDIUM | predicted RUL ≤ 60, or persistent anomaly with predicted RUL ≤ 100 | plan inspection, monitor more often | P3, due before `lower bound − 10` |
| LOW | predicted RUL ≤ 100, or a persistent anomaly the RUL model does not confirm | monitor / watch | none (logged) |
| NORMAL | otherwise | — | none |

An anomaly on its own (RUL model still says > 100 cycles) is only a LOW watch item. The early-life anomaly detector fires earlier than the RUL model, and on validation engines requiring the RUL model's agreement cut work orders on healthy cycles 2–12× with no loss of near-failure detection.

CRITICAL is decided by RUL alone. The classifier answers "fails within 30 cycles?" and is ≈ 1.0 for every engine under 30, so it cannot separate 15 cycles left from 28; it supports HIGH and the disagreement checks instead.

Two classifier thresholds, two jobs: the **F2-tuned threshold** from section 7 (recall-first, typically 0.05–0.16) drives the `fail_alert` early-warning flag and the reported classifier metrics. The agent's HIGH rule uses **P ≥ 0.5**, so HIGH means "more likely than not to fail within 30 cycles". The agent's recall-first behaviour comes from the RUL-based tiers and MEDIUM.

**Human review (exceptions only)** is requested when: the calibrated interval is wider than the review threshold from section 7 (the 10% most uncertain likely-order cases), the classifier and the RUL model disagree, fewer than 15 cycles of history exist, recent data was imputed, the operating condition was never seen in training, or a sensor is faulty. Such work orders get status `NEEDS_REVIEW` instead of `OPEN`; CRITICAL orders are always dispatched (`OPEN`) and only flagged for review.

**Sensor fault ≠ machine fault.** A dead, stuck or out-of-range sensor (section 3) raises a separate `CHECK_SENSOR` ticket ("inspect / replace the sensor"), and any maintenance order on that machine goes to review, because its predictions rest on bad data.

**Order lifecycle (per engine):**
- one open order per engine; a higher severity raises a new order that supersedes the old one;
- **hysteresis:** an order is downgraded (or closed, if no order is needed) only after 5 consecutive calmer cycles, so one noisy cycle never cancels it;
- **review timeout:** a `NEEDS_REVIEW` order nobody handles within 10 cycles is flagged `review_overdue` (escalate to a supervisor);
- **data-trust re-check:** if a sensor fault, an unseen operating condition or filled-in data appears on a machine whose order is already open, that order switches to `NEEDS_REVIEW` with the new reason (CRITICAL stays dispatched, flagged). Ordinary changes in uncertainty don't re-open review.
""")

code(r"""
AGENT_CFG = {
    "fail_window": CFG["fail_window"],
    "critical_rul_lower": 15,
    "high_rul": 30, "high_p_fail": 0.50,
    "medium_rul": 60, "low_rul": 100,
    "review_interval_width": 30,  # replaced by the production bundle's CV-derived review width when it exists
    "review_min_cycles": 15,
    "conflict_p_fail": 0.50, "conflict_rul": 80, "conflict_low_p_fail": 0.10,
    "safety_buffer": 10,
    "deescalate_cycles": 5,       # lower severity must hold this many consecutive cycles before an order is downgraded or closed
    "review_timeout_cycles": 10,  # a NEEDS_REVIEW order not handled within this many cycles is flagged overdue (escalate to a supervisor)
}
SEVERITIES = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
# review reasons that mean the data behind an order can't be trusted (re-checked on already-open orders)
DATA_TRUST_REASONS = ("sensor fault", "operating condition not seen", "recent sensor data was imputed")
PRIORITY = {"CRITICAL": "P1", "HIGH": "P2", "MEDIUM": "P3"}
ACTIONS = {
    "CRITICAL": "Remove from service now; inspect and overhaul the {component} before the next operation.",
    "HIGH": "Schedule {component} maintenance before the due cycle; reserve parts and a maintenance slot.",
    "MEDIUM": "Plan an inspection of the {component} and increase monitoring frequency.",
}


class TriageAgent:
    version = "rules-1.0"

    def __init__(self, cfg, subset, model_version, fault_modes=1):
        self.cfg, self.subset, self.model_version = cfg, subset, model_version
        self.component = "HPC module" if fault_modes == 1 else "HPC and fan modules"
        self.open_orders, self.work_orders, self.log = {}, [], []
        self.calm = {}                                   # consecutive cycles below the open order's severity
        self.sensor_tickets, self.open_sensor_tickets = [], {}

    def assess(self, r):
        c = self.cfg
        rules = [
            ("CRITICAL", r["rul_lo"] <= c["critical_rul_lower"], f"lower-bound RUL {r['rul_lo']:.0f} <= {c['critical_rul_lower']}"),
            ("HIGH", r["rul_pred"] <= c["high_rul"], f"predicted RUL {r['rul_pred']:.0f} <= {c['high_rul']}"),
            ("HIGH", r["p_fail"] >= c["high_p_fail"], f"P(fail within {c['fail_window']}) {r['p_fail']:.2f} >= {c['high_p_fail']}"),
            ("MEDIUM", r["rul_pred"] <= c["medium_rul"], f"predicted RUL {r['rul_pred']:.0f} <= {c['medium_rul']}"),
            # an anomaly raises MEDIUM only when the RUL model agrees the machine is past healthy (RUL <= low_rul);
            # otherwise it is a LOW watch item (validation: same near-failure recall, 2-12x fewer orders on healthy cycles)
            ("MEDIUM", bool(r["anomaly_persistent"]) and r["rul_pred"] <= c["low_rul"],
             f"persistent anomaly (score {r['anomaly_score']:.3f}), RUL {r['rul_pred']:.0f} <= {c['low_rul']}"),
            ("LOW", bool(r["anomaly_persistent"]) and r["rul_pred"] > c["low_rul"],
             f"persistent anomaly not confirmed by RUL model (watch)"),
            ("LOW", r["rul_pred"] <= c["low_rul"], f"predicted RUL {r['rul_pred']:.0f} <= {c['low_rul']}"),
        ]
        fired = [(sev, why) for sev, hit, why in rules if hit]
        severity = max((s for s, _ in fired), key=SEVERITIES.index, default="NORMAL")
        review = []
        if r["rul_hi"] - r["rul_lo"] > c["review_interval_width"]:
            review.append(f"uncertain: interval {r['rul_lo']:.0f}-{r['rul_hi']:.0f} wider than {c['review_interval_width']}")
        if r["p_fail"] >= c["conflict_p_fail"] and r["rul_pred"] > c["conflict_rul"]:
            review.append("models disagree: high failure probability but long predicted RUL")
        if r["rul_pred"] <= c["high_rul"] and r["p_fail"] < c["conflict_low_p_fail"]:
            review.append("models disagree: short predicted RUL but low failure probability")
        if r["cycles_observed"] < c["review_min_cycles"]:
            review.append(f"only {int(r['cycles_observed'])} cycles of history")
        if bool(r["recent_imputed"]):
            review.append("recent sensor data was imputed")
        if bool(r.get("recent_unseen", False)):
            review.append("operating condition not seen in training (nearest known condition used)")
        if bool(r.get("sensor_fault", False)):
            review.append(f"sensor fault: {r.get('sensor_fault_detail', '')}")
        return {"severity": severity, "reasons": [w for _, w in fired], "review_reasons": review}

    def _work_order(self, r, d, supersedes):
        sev = d["severity"]
        due = 0 if sev == "CRITICAL" else max(0, int(np.floor(r["rul_lo"])) - self.cfg["safety_buffer"])
        status = "OPEN" if sev == "CRITICAL" or not d["review_reasons"] else "NEEDS_REVIEW"
        return {
            "work_order_id": f"WO-{self.subset}-{int(r['unit']):03d}-{int(r['cycle']):04d}", "type": "MAINTENANCE",
            "created_at": now_iso(), "subset": self.subset, "unit": int(r["unit"]), "at_cycle": int(r["cycle"]),
            "severity": sev, "priority": PRIORITY[sev], "status": status,
            "human_review": bool(d["review_reasons"]),
            "predicted_rul": round(float(r["rul_pred"]), 1), "rul_lower": round(float(r["rul_lo"]), 1),
            "rul_upper": round(float(r["rul_hi"]), 1), "p_fail_window": round(float(r["p_fail"]), 3),
            "anomaly": bool(r["anomaly_persistent"]), "due_within_cycles": due,
            "recommended_action": ACTIONS[sev].format(component=self.component),
            "reasons": d["reasons"], "review_reasons": d["review_reasons"], "supersedes": supersedes,
            "model_version": self.model_version, "agent_version": self.version,
        }

    def _sensor_ticket(self, r):
        # a broken sensor is a data problem, not a machine problem: separate ticket per machine, re-raised only
        # when another sensor joins the fault list
        unit, detail = int(r["unit"]), str(r.get("sensor_fault_detail", ""))
        faulty = {p.split(" ")[0] for p in detail.split("; ") if p}
        open_t = self.open_sensor_tickets.get(unit)
        if open_t is not None and faulty <= set(open_t["sensors"]):
            return None
        faulty |= set(open_t["sensors"]) if open_t is not None else set()
        t = {"ticket_id": f"SQ-{self.subset}-{unit:03d}-{int(r['cycle']):04d}", "type": "CHECK_SENSOR",
             "created_at": now_iso(), "subset": self.subset, "unit": unit, "at_cycle": int(r["cycle"]),
             "priority": "P3", "status": "OPEN", "sensors": sorted(faulty), "detail": detail,
             "recommended_action": f"Inspect / replace sensor(s) {', '.join(sorted(faulty))} ({detail}). "
                                   "Predictions for this machine are unreliable until fixed.",
             "agent_version": self.version}
        if open_t is not None:
            open_t["status"] = "SUPERSEDED"
        self.open_sensor_tickets[unit] = t
        self.sensor_tickets.append(t)
        return t["ticket_id"]

    def process(self, r):
        d = self.assess(r)
        sev, unit, cycle = d["severity"], int(r["unit"]), int(r["cycle"])
        rank = SEVERITIES.index
        prev = self.open_orders.get(unit)
        self.calm[unit] = self.calm.get(unit, 0) + 1 if prev is not None and rank(sev) < rank(prev["severity"]) else 0
        event, wo_id = "none", None
        if sev in PRIORITY and (prev is None or rank(sev) > rank(prev["severity"])):
            if prev is not None:
                prev["status"] = "SUPERSEDED"
            wo = self._work_order(r, d, prev["work_order_id"] if prev else None)
            self.open_orders[unit] = wo
            self.work_orders.append(wo)
            event, wo_id = ("created" if prev is None else "escalated"), wo["work_order_id"]
        elif prev is not None and self.calm[unit] >= self.cfg["deescalate_cycles"]:
            # hysteresis: only after K consecutive calmer cycles is the order downgraded, or closed if no order is needed
            prev["closed_at_cycle"] = cycle
            if sev in PRIORITY:
                prev["status"] = "SUPERSEDED"
                wo = self._work_order(r, d, prev["work_order_id"])
                self.open_orders[unit] = wo
                self.work_orders.append(wo)
                event, wo_id = "downgraded", wo["work_order_id"]
            else:
                prev["status"] = "CLOSED_RECOVERED"
                del self.open_orders[unit]
                event, wo_id = "closed", prev["work_order_id"]
            self.calm[unit] = 0
        elif prev is not None:
            event, wo_id = "order_already_open", prev["work_order_id"]
        elif sev == "LOW":
            event = "monitor"
        # data-trust problems that appear after an order was raised: the order rests on suspect data, so it goes
        # to human review (CRITICAL stays dispatched, flagged). Uncertainty changes don't re-open review.
        review_requested = False
        cur = self.open_orders.get(unit)
        trust = [x for x in d["review_reasons"] if x.startswith(DATA_TRUST_REASONS)]
        if cur is not None and event == "order_already_open" and trust:
            new = [x for x in trust if x not in cur["review_reasons"]]
            if new:
                cur["review_reasons"] = cur["review_reasons"] + new
                cur["human_review"] = True
                cur.setdefault("review_requested_at_cycle", cycle)
                if cur["status"] == "OPEN" and cur["severity"] != "CRITICAL":
                    cur["status"] = "NEEDS_REVIEW"
                review_requested = True
        overdue = False
        if (cur is not None and cur["status"] == "NEEDS_REVIEW" and not cur.get("review_overdue")
                and cycle - cur["at_cycle"] >= self.cfg["review_timeout_cycles"]):
            cur["review_overdue"], cur["review_overdue_at_cycle"], overdue = True, cycle, True
        ticket = self._sensor_ticket(r) if bool(r.get("sensor_fault", False)) else None
        entry = {"unit": unit, "cycle": cycle, "severity": sev, "event": event, "work_order_id": wo_id,
                 "rul_pred": float(r["rul_pred"]), "rul_lo": float(r["rul_lo"]), "rul_hi": float(r["rul_hi"]),
                 "p_fail": float(r["p_fail"]), "anomaly": bool(r["anomaly_persistent"]),
                 "human_review": bool(d["review_reasons"]), "review_requested": review_requested,
                 "review_overdue": overdue, "sensor_ticket": ticket,
                 "reasons": "; ".join(d["reasons"]), "review_reasons": "; ".join(d["review_reasons"])}
        self.log.append(entry)
        return entry

    def triage(self, frame):
        return pd.DataFrame([self.process(r) for _, r in frame.sort_values(["unit", "cycle"]).iterrows()])


def true_severity(rul, c=AGENT_CFG):
    if rul <= c["critical_rul_lower"]:
        return "CRITICAL"
    if rul <= c["high_rul"]:
        return "HIGH"
    if rul <= c["medium_rul"]:
        return "MEDIUM"
    if rul <= c["low_rul"]:
        return "LOW"
    return "NORMAL"


def wilson_low(k, n, z=1.96):
    # lower end of the 95% Wilson interval: what "k of n caught" supports with a small n
    if n == 0:
        return float("nan")
    p = k / n
    return float((p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n))


def evaluate_agent(decisions, truth, c=AGENT_CFG):
    d = decisions.merge(truth[["unit", "rul"]], on="unit")
    d["true_severity"] = d.rul.map(lambda x: true_severity(x, c))
    has_order = d.severity.isin(list(PRIORITY))
    high_plus = d.severity.isin(["HIGH", "CRITICAL"])
    urgent = d.rul <= c["high_rul"]
    confusion = pd.crosstab(d.true_severity, d.severity).reindex(index=SEVERITIES, columns=SEVERITIES, fill_value=0)
    summary = {
        "engines": int(len(d)), "work_orders": int(has_order.sum()),
        "urgent_engines": int(urgent.sum()),
        "urgent_high_or_critical": int((urgent & high_plus).sum()),
        "urgent_high_or_critical_ci95_low": wilson_low(int((urgent & high_plus).sum()), int(urgent.sum())),
        "urgent_caught": int((urgent & has_order).sum()),
        "urgent_missed": int((urgent & ~has_order).sum()),
        "false_alarms_true_rul_gt_low": int(((d.rul > c["low_rul"]) & has_order).sum()),
        "exact_severity_match_pct": float((d.true_severity == d.severity).mean() * 100),
        "review_rate_pct": float(d.loc[has_order, "human_review"].mean() * 100) if has_order.any() else 0.0,
    }
    return summary, confusion, d
""")

code(r"""
AGENT_CFG["review_interval_width"] = round(b_c["review_width"], 1)
print(f"review threshold: interval wider than {AGENT_CFG['review_interval_width']} cycles")

# the current fleet = every test engine at its latest observed cycle
agent = TriageAgent(AGENT_CFG, SUBSET, model_version=f"{REGISTRY_NAME}/v{PRODUCTION_VERSION}",
                    fault_modes=CFG["fault_modes"])
decisions = agent.triage(fleet)
work_orders = agent.work_orders
agent_summary, agent_confusion, agent_eval = evaluate_agent(decisions, fleet)

print(f"{len(work_orders)} work orders from {len(fleet)} engines")
display(decisions.severity.value_counts().reindex(SEVERITIES, fill_value=0).to_frame("engines"))
display(pd.Series(agent_summary).to_frame("agent vs true RUL"))
print("rows = true severity (from true RUL), columns = agent severity")
display(agent_confusion)
print("example work order:")
print(json.dumps(sorted(work_orders, key=lambda w: SEVERITIES.index(w["severity"]), reverse=True)[0], indent=2))
""", "train")

md(r"""
## 10b. Fault-injection demo: sensor faults and unseen conditions

The C-MAPSS data has no broken sensors, so the data-quality gate never fires on it (checked: zero flags on all train and test data). To show it working, four real test engines are broken in four realistic ways over their last cycles, then run through the same pipeline and agent:
1. a sensor **stuck** at one value for 25 cycles;
2. a sensor reading far **out of range** for 5 cycles;
3. a sensor that **dies** for its last 20 cycles (missing readings get forward-filled, so it then looks stuck);
4. a switch to an **operating condition never seen** in training for 10 cycles.

Expected behaviour: the gate flags each one; sensor faults raise a `CHECK_SENSOR` ticket; any maintenance order on those machines goes to human review.
""")

code(r"""
def inject_faults(raw, prep, units):
    raw = raw.copy()
    sensors, plan = prep["sensors"], []
    hi, lo = pd.DataFrame(prep["sensor_range"]["hi"]).T, pd.DataFrame(prep["sensor_range"]["lo"]).T
    last = lambda u, n: raw.index[raw.unit == u][-n:]
    u_stuck, u_range, u_dead, u_unseen = units
    s = next((x for x in ["s11", "s4", "s7"] if x in prep["stuck_sensors"]), prep["stuck_sensors"][0])
    rows = last(u_stuck, 25)
    raw.loc[rows, s] = raw.at[rows[0], s]
    plan.append((u_stuck, f"{s} stuck", int(raw.at[rows[0], "cycle"])))
    s = "s4" if "s4" in sensors else sensors[0]
    rows = last(u_range, 5)
    raw.loc[rows, s] = float((hi[s] + (hi[s] - lo[s])).max())
    plan.append((u_range, f"{s} out of range", int(raw.at[rows[0], "cycle"])))
    s = next((x for x in ["s7", "s12", "s2"] if x in prep["stuck_sensors"]), prep["stuck_sensors"][-1])
    rows = last(u_dead, 20)
    raw.loc[rows, s] = np.nan
    plan.append((u_dead, f"{s} dead", int(raw.at[rows[0], "cycle"])))
    rows = last(u_unseen, 10)
    raw.loc[rows, OPS] = [30.0, 0.5, 80.0]
    plan.append((u_unseen, "unseen operating condition (30 / 0.5 / 80)", int(raw.at[rows[0], "cycle"])))
    return raw, plan


demo_units = [int(u) for u in fleet.sort_values("cycle").unit.tail(4)]  # longest test histories
raw_faulty, fault_plan = inject_faults(raw_test[raw_test.unit.isin(demo_units)], b_c["prep"], demo_units)
faulty = add_labels(basic_clean(raw_faulty), CFG, rul_final[rul_final.unit.isin(demo_units)])
P_faulty = predict_bundle(b_c, faulty)
demo_agent = TriageAgent(AGENT_CFG, SUBSET, model_version=f"{REGISTRY_NAME}/v{PRODUCTION_VERSION}",
                         fault_modes=CFG["fault_modes"])
demo_log = demo_agent.triage(P_faulty)
clean_decision = decisions.set_index("unit")

rows = []
for u, what, start in fault_plan:
    p, lg = P_faulty[P_faulty.unit == u], demo_log[demo_log.unit == u]
    flagged = p[p.sensor_fault | p.unseen_condition]
    gate = sorted({part for d in p.loc[p.sensor_fault, "sensor_fault_detail"] for part in d.split("; ") if part}
                  | ({"unseen operating condition"} if p.unseen_condition.any() else set()))
    order = demo_agent.open_orders.get(u)
    rows.append({"unit": u, "injected": what, "from_cycle": start, "detected": bool(len(flagged)),
                 "first_flagged_cycle": int(flagged.cycle.min()) if len(flagged) else None,
                 "gate_says": "; ".join(gate),
                 "check_sensor_tickets": len([t for t in demo_agent.sensor_tickets if t["unit"] == u]),
                 "severity_clean_data": clean_decision.at[u, "severity"], "severity_faulty_data": lg.severity.iloc[-1],
                 "rul_clean_data": round(float(clean_decision.at[u, "rul_pred"]), 1),
                 "rul_faulty_data": round(float(lg.rul_pred.iloc[-1]), 1),
                 "order_status": order["status"] if order else "no order",
                 "order_flagged_for_review": bool(order["human_review"]) if order else False,
                 "review_reasons": lg.review_reasons.iloc[-1]})
fault_demo = pd.DataFrame(rows)
display(fault_demo)
""", "train")

md(r"""
## 11. Fleet replay and comparison with today's maintenance practice

Every validation engine (they run all the way to failure, and the champion never trained on them) is scored by the champion and streamed through **one stateful agent, cycle by cycle**, like a live fleet. This evaluates the agent over time instead of at a single snapshot:
- **warning lead time:** true cycles left when the first order, the first HIGH+ order and the first CRITICAL order are raised;
- **stability:** orders per engine, downgrades / closures (hysteresis at work), overdue reviews.

The same replay is then compared with the two policies the problem statement describes:
- **reactive:** run to failure, so every failure is unplanned;
- **fixed interval T:** service every engine at cycle T, with T set from the *non-validation* engines' history (the largest T with no failures in that history, and the T allowing 5% failures);
- **ours:** service at the due cycle of each engine's first HIGH or CRITICAL order. If the engine's life ends first, it counts as an unplanned failure.
""")

code(r"""
def fleet_replay(P, model_version):
    agent_ = TriageAgent(AGENT_CFG, SUBSET, model_version=model_version, fault_modes=CFG["fault_modes"])
    log = pd.DataFrame([agent_.process(r) for _, r in P.sort_values(["unit", "cycle"]).iterrows()])
    return log.merge(P[["unit", "cycle", "rul"]], on=["unit", "cycle"]), agent_


def engine_stats(log, agent_, lives):
    wo = pd.DataFrame(agent_.work_orders)
    rows = []
    for u, g in log.groupby("unit"):
        raised = g[g.event.isin(["created", "escalated", "downgraded"])]
        first = lambda sevs: raised.loc[raised.severity.isin(sevs), "rul"].iloc[0] if raised.severity.isin(sevs).any() else np.nan
        hc = wo[(wo.unit == u) & wo.severity.isin(["HIGH", "CRITICAL"])] if len(wo) else wo
        service = int((hc.at_cycle + hc.due_within_cycles).min()) if len(hc) else None
        life = int(lives[u])
        failed = service is None or service >= life
        rows.append({"unit": int(u), "life": life, "first_order_rul": first(["MEDIUM", "HIGH", "CRITICAL"]),
                     "first_high_rul": first(["HIGH", "CRITICAL"]), "first_critical_rul": first(["CRITICAL"]),
                     "orders": len(raised), "downgrades_or_closures": int(g.event.isin(["downgraded", "closed"]).sum()),
                     "reviews_overdue": int(g.review_overdue.sum()), "service_cycle": service, "failed": failed,
                     "cycles_left_at_service": None if failed else life - service})
    return pd.DataFrame(rows)


def policy_eval(service, lives):
    failed = [s is None or s >= l for s, l in zip(service, lives)]
    used = [1.0 if f else s / l for s, l, f in zip(service, lives, failed)]
    wasted = [0 if f else l - s for s, l, f in zip(service, lives, failed)]
    return {"unplanned_failures": int(sum(failed)), "engines": len(lives),
            "life_used_pct": float(np.mean(used) * 100), "mean_cycles_wasted": float(np.mean(wasted))}


val_lives = Pv_c.groupby("unit").cycle.max()
replay_log, replay_agent = fleet_replay(Pv_c, f"{REGISTRY_NAME}/v{CHAMPION_VERSION}")
fleet_stats = engine_stats(replay_log, replay_agent, val_lives)
fleet_replay_summary = {
    "engines": int(len(fleet_stats)),
    "median_rul_at_first_order": float(fleet_stats.first_order_rul.median()),
    "median_rul_at_first_high": float(fleet_stats.first_high_rul.median()),
    "engines_critical_before_failure_pct": float(fleet_stats.first_critical_rul.notna().mean() * 100),
    "mean_orders_per_engine": float(fleet_stats.orders.mean()),
    "engines_with_downgrade_or_closure": int((fleet_stats.downgrades_or_closures > 0).sum()),
    "reviews_overdue": int(fleet_stats.reviews_overdue.sum()),
    "min_cycles_left_at_service": float(fleet_stats.cycles_left_at_service.min()),
    "median_cycles_left_at_service": float(fleet_stats.cycles_left_at_service.median()),
}

hist_lives = train_df[~train_df.unit.isin(SPLITS["val"])].groupby("unit").cycle.max()
t_safe, t_5pct = int(hist_lives.min()) - 1, int(np.quantile(hist_lives, 0.05))
lives = [int(val_lives[u]) for u in fleet_stats.unit]
policy_comparison = pd.DataFrame({
    "reactive (run to failure)": policy_eval([None] * len(lives), lives),
    f"fixed interval T={t_safe} (0% failures in history)": policy_eval([t_safe] * len(lives), lives),
    f"fixed interval T={t_5pct} (5% failures in history)": policy_eval([t_5pct] * len(lives), lives),
    "predictive agent (ours)": policy_eval([None if pd.isna(s) else int(s) for s in fleet_stats.service_cycle], lives),
}).T
display(pd.Series(fleet_replay_summary).round(1).to_frame("fleet replay (validation engines)"))
display(policy_comparison.round(1))

replay_unit = int(val_lives.idxmax())  # longest-lived validation engine, plotted below
events = replay_log[(replay_log.unit == replay_unit) & replay_log.event.isin(["created", "escalated", "downgraded", "closed"])]
display(events[["cycle", "rul", "severity", "event", "rul_pred", "rul_lo", "p_fail", "human_review", "reasons"]]
        .rename(columns={"rul": "true_rul"}))

e = Pv_c[Pv_c.unit == replay_unit]
fig, ax = plt.subplots(figsize=(12, 4))
ax.fill_between(e.cycle, e.rul_lo, e.rul_hi, alpha=0.2, label="10-90% interval")
ax.plot(e.cycle, e.rul, "k-", lw=1, label="true RUL")
ax.plot(e.cycle, e.rul_pred, lw=1.5, label="predicted RUL")
colors = {"MEDIUM": "gold", "HIGH": "orange", "CRITICAL": "red"}
for _, ev in events.iterrows():
    label = ev.severity if ev.event != "closed" else "closed"
    ax.axvline(ev.cycle, color=colors.get(ev.severity, "green"), ls="--", lw=1.2)
    ax.text(ev.cycle, ax.get_ylim()[1] * 0.95, f" {label}\n true RUL {ev.rul}", fontsize=8, va="top")
ax.set(xlabel="cycle", ylabel="RUL (cycles)", title=f"streaming replay · validation engine {replay_unit}")
ax.legend(loc="lower left")
plots.append(savefig(fig, "replay"))

# today's practice vs ours: life used against unplanned failures
fig, ax = plt.subplots(figsize=(7, 4))
grid = range(int(min(lives)) - 30, int(max(lives)) + 1, 5)
curve = [policy_eval([t] * len(lives), lives) for t in grid]
ax.plot([c["life_used_pct"] for c in curve], [c["unplanned_failures"] for c in curve], "-", color="grey",
        label="fixed interval (sweep of T)")
ours = policy_comparison.loc["predictive agent (ours)"]
ax.scatter([ours.life_used_pct], [ours.unplanned_failures], color="red", zorder=3, s=60, label="predictive agent (ours)")
ax.set(xlabel="useful life used before service (%)", ylabel=f"unplanned failures (of {len(lives)})",
       title=f"{SUBSET}: maintenance policies on validation engines")
ax.legend()
plots.append(savefig(fig, "policy_comparison"))
""", "train")

md(r"""
## 12. Monitoring outputs and export

Everything the dashboard and the review need, in `/kaggle/working/artifacts/<SUBSET>` plus the tracking store, zipped into `/kaggle/working/<SUBSET>_outputs.zip` (download it from the notebook's *Output* tab).

| File | Contents |
|---|---|
| `production_bundle.joblib` | the deployed bundle (preprocessing + all models + thresholds): the full refit, or the champion if `full_refit` is off |
| `champion_bundle.joblib` | the validated champion from the promotion step |
| `fleet_status.csv` | latest production prediction and agent decision per test engine |
| `predictions_val.csv` | per-cycle champion predictions on validation engines |
| `predictions_test.csv` | per-cycle production predictions on test engines |
| `work_orders.json` / `.csv` | generated work orders |
| `replay_log.csv` | cycle-by-cycle agent decisions for every validation engine (fleet replay) |
| `fleet_engine_stats.csv` | per validation engine: warning lead times, orders, service cycle, failed or not |
| `sensor_tickets.json` | CHECK_SENSOR tickets raised on the test fleet (normally none) |
| `fault_injection.csv` | the section 10b demo: injected faults and what the gate and agent did |
| `monitoring.json` | CV results, model history (v1/v2/v3), drift (PSI), promotion decision, champion vs production, agent evaluation, configs, splits |
| `plots/*.png` | evaluation charts |
| `tracking/` | MLflow SQLite store (or JSON fallback) with runs and the model registry |
""")

code(r"""
shutil.copy(OUT / "models" / f"bundle_{champion['bundle']['label']}.joblib", OUT / "champion_bundle.joblib")
shutil.copy(OUT / "models" / f"bundle_{b_c['label']}.joblib", OUT / "production_bundle.joblib")
save_json(b_c["prep"], OUT / "preprocessing.json")

pred_cols = ["unit", "cycle", "rul", "rul_pred", "rul_lo", "rul_hi", "p_fail", "fail_alert", "anomaly_score",
             "anomaly_persistent"]
Pv_c[pred_cols].to_csv(OUT / "predictions_val.csv", index=False)
Pt_c[pred_cols].to_csv(OUT / "predictions_test.csv", index=False)
agent_eval.rename(columns={"rul": "true_rul"}).to_csv(OUT / "fleet_status.csv", index=False)
save_json(work_orders, OUT / "work_orders.json")
wo_table = pd.DataFrame(work_orders)
if len(wo_table):
    wo_table["reasons"] = wo_table.reasons.str.join("; ")
    wo_table["review_reasons"] = wo_table.review_reasons.str.join("; ")
wo_table.to_csv(OUT / "work_orders.csv", index=False)
replay_log.to_csv(OUT / "replay_log.csv", index=False)
fleet_stats.to_csv(OUT / "fleet_engine_stats.csv", index=False)
save_json(agent.sensor_tickets, OUT / "sensor_tickets.json")
fault_demo.to_csv(OUT / "fault_injection.csv", index=False)

monitoring = {
    "subset": SUBSET, "seed": SEED, "run_tag": RUN_TAG, "generated_at": now_iso(), "registry_name": REGISTRY_NAME,
    "champion_version": CHAMPION_VERSION, "champion_model": champion["bundle"]["kind"],
    "champion_metrics": m_v, "production_version": PRODUCTION_VERSION, "production_label": b_c["label"],
    "production_engines": len(b_c["units"]), "production_metrics": m_c,
    "refit_comparison": refit_comparison.to_dict(), "test_error_by_band": band_table.to_dict(orient="index"),
    # selection/calibration of the production version; per-version details under "selections"
    "cv_results": b_c["selection"]["cv_results"].to_dict(orient="index"), "cv_classifier": b_c["selection"]["cv_clf"],
    "clf_threshold": b_c["clf_threshold"], "cv_interval": b_c["selection"]["cv_interval"],
    "selections": {s["label"]: {"engines": s["engines"], "model": s["kind"], "clf_threshold": s["clf_threshold"],
                                "cv_results": s["cv_results"].to_dict(orient="index"), "cv_classifier": s["cv_clf"],
                                "cv_interval": s["cv_interval"]}
                   for s in [SEL_V1, globals().get("SEL_V2")] if s is not None},
    "history": history,
    "drift_psi": drift.to_dict(), "promotion": promotion,
    "agent_summary": agent_summary, "agent_confusion": agent_confusion.to_dict(),
    "fleet_replay": fleet_replay_summary, "policy_comparison": policy_comparison.to_dict(orient="index"),
    "policy_fixed_intervals": {"t_safe": t_safe, "t_5pct": t_5pct},
    "fault_injection": fault_demo.to_dict(orient="records"),
    "config": CFG, "agent_config": AGENT_CFG, "splits": SPLITS, "ingest_report": ingest_report,
    "sensors_kept": b_c["prep"]["sensors"], "sensors_dropped": b_c["prep"]["dropped"],
}
save_json(monitoring, OUT / "monitoring.json")
save_json({"config": CFG, "agent_config": AGENT_CFG}, OUT / "config.json")

with tracker.run(f"{SUBSET}_agent", subset=SUBSET, stage="agent", version=PRODUCTION_VERSION):
    tracker.log_params(AGENT_CFG)
    tracker.log_metrics(prefix("agent", agent_summary))
    for name in ["work_orders.json", "fleet_status.csv", "monitoring.json"]:
        tracker.log_artifact(OUT / name)

# run_all_subsets.ipynb sets SKIP_ZIP and makes one combined zip at the end instead
zip_path = "skipped (combined zip made by run_all_subsets)"
if not globals().get("SKIP_ZIP", False):
    bundle_dir = WORK / f"{SUBSET}_outputs"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    shutil.copytree(OUT, bundle_dir / "artifacts")
    shutil.copytree(WORK / "tracking", bundle_dir / "tracking")
    zip_path = shutil.make_archive(str(bundle_dir), "zip", root_dir=bundle_dir)
    shutil.rmtree(bundle_dir)

print(f"champion: {REGISTRY_NAME} v{CHAMPION_VERSION} | production: v{PRODUCTION_VERSION} "
      f"({b_c['kind']}, {len(b_c['units'])} engines)")
print(f"production test RMSE {m_c['test_rmse_capped']:.2f} capped / {m_c['test_rmse']:.2f} true | "
      f"NASA score {m_c['test_nasa_score']:.1f} | "
      f"urgent engines caught {agent_summary['urgent_caught']}/{agent_summary['urgent_engines']}")
print("outputs:", zip_path)
""", "train")

md(r"""
## Next steps

- Run another subset by changing `SUBSET`, or all four in one go with `run_all_subsets.ipynb`.
- Add a sequence model (LSTM / GRU on the T4s) as another candidate in section 7 and let the same CV and promotion rules decide.
- Point the monitoring dashboard at the exported `monitoring.json`, `fleet_status.csv` and `work_orders.json`.
""")


ABLATION_CELL = r"""
# ---- ablation: change one design choice at a time, same 5-fold CV on the non-validation training engines ----
# Base model is HistGradientBoosting for every variant, so only the design choice changes; model choice is its own block.
# Metrics on held-out 'test-like' rows (>= 31 cycles observed, true RUL <= 200); no test data is used.
dev = train_df[train_df.unit.isin(SPLITS["initial"] + SPLITS["new"])].reset_index(drop=True)
folds = list(GroupKFold(n_splits=CFG["cv_folds"]).split(dev, groups=dev.unit))
EVAL_CAP = 125  # every variant is scored against the same truth, whatever cap it trained on


def drop_suffix(suffix):
    return lambda cols, sensors: [c for c in cols if not c.endswith(suffix) or c == "cycle"]


VARIANTS = {  # name: (cfg overrides -> new preprocessing, column selector, training cap, model)
    "baseline (current design)": ({}, None, 125, "hgb"),
    "raw sensors only (no rolling features)": ({}, lambda cols, sensors: ["cycle"] + sensors, 125, "hgb"),
    "no trend (slope) features": ({}, lambda cols, sensors: [c for c in cols if "_slope" not in c], 125, "hgb"),
    "no drift-from-own-baseline": ({}, drop_suffix("_drift"), 125, "hgb"),
    "no cycle count": ({}, lambda cols, sensors: [c for c in cols if c != "cycle"], 125, "hgb"),
    "shorter windows (5 / 10 / 5)": ({"roll_window": 5, "slope_window": 10, "ewm_span": 5}, None, 125, "hgb"),
    "longer windows (30 / 50 / 30)": ({"roll_window": 30, "slope_window": 50, "ewm_span": 30}, None, 125, "hgb"),
    "RUL cap 110": ({}, None, 110, "hgb"),
    "RUL cap 140": ({}, None, 140, "hgb"),
    "no RUL cap": ({}, None, 10_000, "hgb"),
    "model: Ridge (linear)": ({}, None, 125, "ridge"),
}
if CFG["condition_norm"]:
    VARIANTS["no per-condition normalisation"] = ({"condition_norm": False}, None, 125, "hgb")
if xgb is not None:
    VARIANTS["model: XGBoost"] = ({}, None, 125, "xgb")
    VARIANTS["model: HistGB + XGBoost average"] = ({}, None, 125, "ens")


def fit_predict(kind, Xtr, ytr, Xva, cap):
    if kind == "ens":
        p = (make_rul_model("hgb").fit(Xtr, ytr).predict(Xva) + make_rul_model("xgb").fit(Xtr, ytr).predict(Xva)) / 2
    else:
        p = make_rul_model(kind).fit(Xtr, ytr).predict(Xva)
    return np.clip(p, 0, cap)


def scores(y, p):
    d_cap = p - np.minimum(y, EVAL_CAP)
    near = y <= 60
    return {"rmse_capped": float(np.sqrt(np.mean(d_cap ** 2))), "rmse_true": float(np.sqrt(np.mean((p - y) ** 2))),
            "rmse_true_rul_le_60": float(np.sqrt(np.mean((p[near] - y[near]) ** 2))) if near.any() else float("nan"),
            "nasa_capped_mean": float(nasa_score(np.minimum(y, EVAL_CAP), p).mean())}


groups = {}
for name, (over, sel, cap_, kind) in VARIANTS.items():
    groups.setdefault(json.dumps(over, sort_keys=True), []).append(name)

oof = {name: [] for name in VARIANTS}
calib = []  # baseline quantile models per fold, for the calibration ablation
t0 = time.time()
for key, names in groups.items():
    cfg_v = {**CFG, **json.loads(key)}
    for k, (tr_idx, va_idx) in enumerate(folds):
        tr, va = dev.iloc[tr_idx], dev.iloc[va_idx]
        prep = fit_prep(tr, cfg_v)
        Ftr, Fva = featurize(tr, prep, cfg_v), featurize(va, prep, cfg_v)
        rows = (Fva.cycle >= CFG["eval_min_cycles"]) & (Fva.rul <= CFG["eval_max_rul"])
        cols_all = feature_cols(prep, cfg_v)
        for name in names:
            _, sel, cap_, kind = VARIANTS[name]
            cols = sel(cols_all, prep["sensors"]) if sel else cols_all
            p = fit_predict(kind, Ftr[cols], np.minimum(Ftr.rul, cap_), Fva.loc[rows, cols], cap_)
            oof[name].append(pd.DataFrame({"fold": k, "rul": Fva.loc[rows, "rul"].to_numpy(), "pred": p}))
            if name == "baseline (current design)":
                y = np.minimum(Ftr.rul, 125)
                calib.append(pd.DataFrame({"fold": k, "rul": Fva.loc[rows, "rul"].to_numpy(),
                                           "lo": make_quantile(0.1).fit(Ftr[cols], y).predict(Fva.loc[rows, cols]),
                                           "hi": make_quantile(0.9).fit(Ftr[cols], y).predict(Fva.loc[rows, cols]),
                                           "pred": p}))
        print(f"[{SUBSET}] {', '.join(names)[:90]}... fold {k} ({time.time() - t0:.0f}s)")

base = pd.concat(oof["baseline (current design)"])
base_fold = base.groupby("fold").apply(lambda d: scores(d.rul.to_numpy(), d.pred.to_numpy())["rmse_capped"])
table = []
for name in VARIANTS:
    o = pd.concat(oof[name])
    per_fold = o.groupby("fold").apply(lambda d: scores(d.rul.to_numpy(), d.pred.to_numpy())["rmse_capped"])
    delta = per_fold - base_fold
    table.append({"variant": name, **scores(o.rul.to_numpy(), o.pred.to_numpy()),
                  "fold_std_rmse_capped": float(per_fold.std()), "delta_vs_baseline": float(delta.mean()),
                  "folds_better_than_baseline": f"{int((delta < 0).sum())}/{len(delta)}"})
ablation = pd.DataFrame(table).set_index("variant").round(3)

# calibration ablation, cross-fitted: each fold's margin comes from the other folds
c = pd.concat(calib)
c["y"] = np.minimum(c.rul, 125)
covered = lambda d, q: ((d.y >= np.minimum(np.clip(d.lo - q, 0, 125), d.pred)) & (d.y <= np.maximum(np.clip(d.hi + q, 0, 125), d.pred)))
raw_cov, cal_cov, near_raw, near_cal = [], [], [], []
for k in sorted(c.fold.unique()):
    others, this = c[(c.fold != k) & (c.rul <= 125)], c[c.fold == k]
    q = conformal_correction(others.y, others.lo, others.hi, CFG["interval_coverage"])
    raw_cov.append(covered(this[this.rul <= 125], 0.0).mean()); cal_cov.append(covered(this[this.rul <= 125], q).mean())
    near = this[this.rul <= 30]
    near_raw.append(covered(near, 0.0).mean()); near_cal.append(covered(near, q).mean())
calibration = pd.Series({"coverage_raw_quantiles": np.mean(raw_cov), "coverage_calibrated": np.mean(cal_cov),
                         "coverage_raw_rul_le_30": np.mean(near_raw), "coverage_calibrated_rul_le_30": np.mean(near_cal),
                         "target": CFG["interval_coverage"]}).round(3)

display(ablation)
display(calibration.to_frame("interval calibration (cross-fitted)"))
ABLATION_OUT = OUT / "ablation"
ABLATION_OUT.mkdir(parents=True, exist_ok=True)
ablation.to_csv(ABLATION_OUT / "ablation.csv")
save_json({"subset": SUBSET, "engines": int(dev.unit.nunique()), "folds": CFG["cv_folds"], "base_model": "hgb",
           "eval_cap": EVAL_CAP, "variants": ablation.reset_index().to_dict(orient="records"),
           "calibration": calibration.to_dict()}, ABLATION_OUT / "ablation.json")
ABLATION_RESULT = {"table": ablation, "calibration": calibration}
"""


def ablation_cells(pipeline_cells):
    # base pipeline cells up to the model definitions (no tracking, no training), then the ablation code
    code_cells = []
    for kind, src, tags in pipeline_cells:
        if kind != "code" or 'ensure("mlflow")' in src or "class Tracker" in src:
            continue
        if "train" in tags:
            break
        code_cells.append(src)
    assert any("def make_rul_model" in s for s in code_cells), "model definitions missing from ablation base"
    payload = json.dumps(code_cells + [ABLATION_CELL.strip("\n")], indent=0)
    assert "'''" not in payload
    cells = [("markdown", r"""
# Ablation study: what each design choice is worth

For each subset, one design choice is removed or changed at a time and scored with the **same 5-fold cross-validation on the non-validation training engines** (paired folds; no test data). The base model is HistGradientBoosting for every variant, so only the design choice changes; model choice is a separate block.

| Block | Variants |
|---|---|
| Features | raw sensors only · no trend (slope) · no drift-from-own-baseline · no cycle count |
| Windows | shorter (5/10/5) · longer (30/50/30) vs current (10/20/10) |
| Target | RUL cap 110 · 140 · none vs 125 |
| Regimes | no per-condition normalisation (FD002 / FD004 only) |
| Model | Ridge · XGBoost · HistGB + XGBoost average |
| Calibration | raw quantile coverage vs conformal-calibrated (cross-fitted), overall and for RUL ≤ 30 |

**Metrics** (held-out "test-like" rows): `rmse_capped` (vs min(true RUL, 125), the same truth for every variant), `rmse_true`, `rmse_true_rul_le_60` (where decisions happen), `nasa_capped_mean`, the spread across folds, and the paired difference vs baseline (`delta_vs_baseline`, negative = better; `folds_better_than_baseline`).

**Run on Kaggle** like the main notebook (dataset attached, GPU, Internet not needed). Roughly 10–20 minutes per subset. Output: `/kaggle/working/ablation_outputs.zip`. Generated from `scripts/build_notebook.py`; edit it there.
""".strip("\n"), [])]
    cells.append(("code", "import json\n\n# pipeline code cells + ablation code, generated from scripts/build_notebook.py\n"
                  "ABLATION_CELLS = json.loads(r'''" + payload + "''')\n"
                  "print(len(ABLATION_CELLS), 'cells loaded')", []))
    cells.append(("code", r"""
import gc, shutil, time, traceback
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
WORK = Path("/kaggle/working") if Path("/kaggle/input").exists() else Path.cwd() / "outputs"
shutil.rmtree(WORK / "artifacts", ignore_errors=True)

ablations = {}
for subset in SUBSETS:
    print(f"\n{'=' * 34} {subset} {'=' * 34}")
    ns = {"__name__": "__main__", "display": display}
    t0 = time.time()
    try:
        for i, src in enumerate(ABLATION_CELLS):
            exec(compile(src.replace('SUBSET = "FD001"', f'SUBSET = "{subset}"'), f"<{subset} cell {i}>", "exec"), ns)
        ablations[subset] = ns["ABLATION_RESULT"]
    except Exception:
        traceback.print_exc()
    finally:
        plt.close("all")
        del ns
        gc.collect()
    print(f"{subset}: {'ok' if subset in ablations else 'FAILED'} in {(time.time() - t0) / 60:.1f} min")
""".strip("\n"), ["train"]))
    cells.append(("markdown", r"""
## Cross-subset summary

`delta_vs_baseline` is the change in capped RMSE (cycles) against the current design, averaged over the 5 paired folds: positive = that change makes things worse (so the design choice it removes is worth that much), negative = it would be an improvement.
""".strip("\n"), []))
    cells.append(("code", r"""
if ablations:
    order = list(dict.fromkeys(v for r in ablations.values() for v in r["table"].index))  # keep the variant grouping
    delta = pd.DataFrame({s: r["table"]["delta_vs_baseline"] for s, r in ablations.items()}).reindex(order)
    rmse = pd.DataFrame({s: r["table"]["rmse_capped"] for s, r in ablations.items()}).reindex(order)
    wins = pd.DataFrame({s: r["table"]["folds_better_than_baseline"] for s, r in ablations.items()}).reindex(order)
    calib = pd.DataFrame({s: r["calibration"] for s, r in ablations.items()})
    print("capped RMSE (cycles), CV on training engines")
    display(rmse)
    print("change vs current design (cycles; + = worse)")
    display(delta.round(2))
    print("folds where the variant beat the current design")
    display(wins)
    print("interval calibration (cross-fitted)")
    display(calib)
    out = WORK / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    rmse.to_csv(out / "ablation_rmse_capped.csv")
    delta.to_csv(out / "ablation_delta_vs_baseline.csv")
    wins.to_csv(out / "ablation_folds_better.csv")
    calib.to_csv(out / "ablation_calibration.csv")
    zip_path = shutil.make_archive(str(WORK / "ablation_outputs"), "zip", root_dir=out)
    print("outputs:", zip_path)
""".strip("\n"), ["train"]))
    return cells


DEEP_DEFS_CELL = r"""
# ---- deep sequence models (PyTorch): 1D-CNN, GRU and TCN on windows of past normalised sensor readings ----
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS (set before CUDA starts)
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
print("torch", torch.__version__, "| device", DEVICE)


class CNN1D(nn.Module):
    # the classic C-MAPSS conv net: stacked 1D convolutions over the window, then mean + last-step pooling
    def __init__(self, c_in, width=64, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(c_in, 32, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(32, width, 5, padding=2), nn.ReLU(),
                                  nn.Conv1d(width, width, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * width, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):  # x: (batch, time, channels)
        h = self.conv(x.transpose(1, 2))
        return self.head(torch.cat([h.mean(-1), h[..., -1]], 1)).squeeze(-1)


class GRUNet(nn.Module):
    # recurrent: two stacked GRU layers, read out at the latest cycle
    def __init__(self, c_in, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(c_in, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)


class TCNBlock(nn.Module):
    # causal dilated convolutions (left padding only) with a residual connection
    def __init__(self, c_in, c_out, dilation, k=3, dropout=0.1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.c1 = nn.Conv1d(c_in, c_out, k, dilation=dilation)
        self.c2 = nn.Conv1d(c_out, c_out, k, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.drop(torch.relu(self.c1(nn.functional.pad(x, (self.pad, 0)))))
        h = self.drop(torch.relu(self.c2(nn.functional.pad(h, (self.pad, 0)))))
        return torch.relu(h + self.skip(x))


class TCN(nn.Module):
    def __init__(self, c_in, width=64, dilations=(1, 2, 4, 8)):
        super().__init__()
        blocks, c = [], c_in
        for d in dilations:  # receptive field 61 cycles >= the 30-cycle window
            blocks.append(TCNBlock(c, width, d))
            c = width
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        return self.head(self.net(x.transpose(1, 2))[..., -1]).squeeze(-1)


NETS = {"cnn": CNN1D, "gru": GRUNet, "tcn": TCN}


def _batches(n, size, rng=None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, size):
        yield idx[i:i + size]


def predict_dl(model, X, cap, batch=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for b in _batches(len(X), batch):
            out.append(model(torch.from_numpy(X[b]).to(DEVICE)).float().cpu().numpy())
    return np.clip(np.concatenate(out) * cap, 0, cap) if out else np.zeros(0)


def fit_dl(kind, Xtr, ytr, Xes, yes, dl, seed=SEED):
    # MSE on the capped target scaled to [0, 1]; early stopping on engines the net does not train on.
    # `seed` sets the initial weights and the batch order, the two sources of run-to-run spread in a net
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = NETS[kind](Xtr.shape[2]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=dl["lr"], weight_decay=dl["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    # the training windows fit in GPU memory (largest subset ~0.1 GB), so batches are gathered on the device
    Xt = torch.from_numpy(Xtr).to(DEVICE)
    yt = torch.from_numpy((ytr / dl["cap"]).astype(np.float32)).to(DEVICE)
    best, best_epoch, bad, t0 = np.inf, 0, 0, time.time()
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for epoch in range(dl["epochs"]):
        model.train()
        for b in _batches(len(Xt), dl["batch"], rng):
            b = torch.from_numpy(b).to(DEVICE)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xt[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        rmse = float(np.sqrt(np.mean((predict_dl(model, Xes, dl["cap"]) - yes) ** 2)))
        sched.step(rmse)
        if rmse < best - 1e-3:
            best, best_epoch, bad = rmse, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= dl["patience"]:
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "epochs_run": epoch + 1, "early_stop_rmse_capped": round(best, 3),
                   "seconds": round(time.time() - t0, 1), "params": sum(p.numel() for p in model.parameters())}
"""

DEEP_RUN_CELL = r"""
# ---- deep models vs gradient boosting: same engines, same 5 CV folds, same truth, then the official test set ----
DL = {"window": 30, "epochs": 60, "patience": 8, "batch": 512, "lr": 1e-3, "weight_decay": 1e-4,
      "es_frac": 0.15, "cap": 125,
      # each net is trained from 3 random starts; the deployable net is their average (a "seed ensemble"),
      # and the per-seed scores show how much of a difference is luck of the initialisation
      "seeds": [42, 1, 2], **globals().get("DL_OVERRIDES", {})}
EVAL_CAP = 125  # every model is scored against min(true RUL, 125)
DEEP = list(NETS)
TREES = ["hgb"] + (["xgb"] if xgb is not None else [])
dev = train_df[train_df.unit.isin(SPLITS["initial"] + SPLITS["new"])].reset_index(drop=True)  # = v2's training set
folds = list(GroupKFold(n_splits=CFG["cv_folds"]).split(dev, groups=dev.unit))  # identical to the ablation folds


def windows(dz, sensors, w):
    # one window per row: the last w cycles up to and including that row (past only), left-padded with the
    # engine's first reading; channels = the normalised sensors + cycle / 100 (the trees get cycle too)
    blocks = []
    for _, g in dz.groupby("unit", sort=True):
        y = np.column_stack([g[sensors].to_numpy(np.float32), g.cycle.to_numpy(np.float32) / 100.0])
        y = np.vstack([np.repeat(y[:1], w - 1, axis=0), y])
        blocks.append(np.lib.stride_tricks.sliding_window_view(y, w, axis=0).transpose(0, 2, 1))
    return np.ascontiguousarray(np.concatenate(blocks), dtype=np.float32)


def check_window_causality(df, prep, w, n_units=4):
    # a row's window must not change when the engine's future cycles are removed
    full = windows(apply_prep(df, prep), prep["sensors"], w)
    z = apply_prep(df, prep)
    for u in z.unit.unique()[:n_units]:
        g = df[df.unit == u]
        cut = int(g.cycle.max() * 0.6)
        part = windows(apply_prep(g[g.cycle <= cut], prep), prep["sensors"], w)
        row = np.flatnonzero((z.unit == u).to_numpy() & (z.cycle == cut).to_numpy())[0]
        assert np.allclose(part[-1], full[row]), f"window leak on engine {u}"


def test_like_mask(F):
    return ((F.cycle >= CFG["eval_min_cycles"]) & (F.rul <= CFG["eval_max_rul"])).to_numpy()


def fit_all(tr, tr_dl, es, targets):
    # trees train on `tr`; nets train on `tr_dl` and early-stop on `es`; one preprocessing fit on `tr` for both.
    # Returns predictions on every target frame (rows in apply_prep order) and the nets' training info.
    prep = fit_prep(tr, CFG)
    cols, sensors, w = feature_cols(prep, CFG), prep["sensors"], DL["window"]
    Ftr = featurize(tr, prep, CFG)
    F = {k: featurize(d, prep, CFG) for k, d in targets.items()}
    preds = {k: {} for k in targets}
    for kind in TREES:
        m = make_rul_model(kind).fit(Ftr[cols], np.minimum(Ftr.rul, DL["cap"]))
        for k in targets:
            preds[k][kind] = np.clip(m.predict(F[k][cols]), 0, DL["cap"])
    Ztr, Zes = apply_prep(tr_dl, prep), apply_prep(es, prep)
    Xtr, Xes = windows(Ztr, sensors, w), windows(Zes, sensors, w)
    es_rows = ((Zes.cycle >= CFG["eval_min_cycles"]) & (Zes.rul <= CFG["eval_max_rul"])).to_numpy()
    if not es_rows.any():
        es_rows[:] = True
    Xes, yes = Xes[es_rows], np.minimum(Zes.rul.to_numpy()[es_rows], EVAL_CAP).astype(float)
    Xt = {k: windows(apply_prep(d, prep), sensors, w) for k, d in targets.items()}
    info, ytr = {}, np.minimum(Ztr.rul.to_numpy(), DL["cap"]).astype(np.float32)
    for kind in DEEP:
        for s in DL["seeds"]:
            model, info[f"{kind} [seed {s}]"] = fit_dl(kind, Xtr, ytr, Xes, yes, DL, seed=s)
            for k in targets:
                preds[k][f"{kind} [seed {s}]"] = predict_dl(model, Xt[k], DL["cap"])
            del model
        for k in targets:  # the net as deployed: average of its seeds
            preds[k][kind] = np.mean([preds[k][f"{kind} [seed {s}]"] for s in DL["seeds"]], axis=0)
    for k in targets:  # ensembles: trees + each net, and all three nets together
        p = preds[k]
        for kind in DEEP:
            p[f"hgb + {kind}"] = (p["hgb"] + p[kind]) / 2
        p["cnn + gru + tcn"] = np.mean([p[kind] for kind in DEEP], axis=0)
        p["hgb + cnn + gru + tcn"] = (p["hgb"] + p["cnn + gru + tcn"]) / 2
    return preds, F, info, prep


def scores(y, p):
    near = y <= 60
    return {"rmse_capped": float(np.sqrt(np.mean((p - np.minimum(y, EVAL_CAP)) ** 2))),
            "rmse_true": float(np.sqrt(np.mean((p - y) ** 2))),
            "rmse_true_rul_le_60": float(np.sqrt(np.mean((p[near] - y[near]) ** 2))) if near.any() else float("nan"),
            "nasa_capped_mean": float(nasa_score(np.minimum(y, EVAL_CAP), p).mean())}


check_window_causality(dev, fit_prep(dev, CFG), DL["window"])
print(f"[{SUBSET}] window causality check passed; {dev.unit.nunique()} engines, {len(folds)} folds, window {DL['window']}")

# 1) cross-validation on the training engines (paired folds; nets early-stop on 15% of each fold's training engines)
oof, fit_info, t0 = [], [], time.time()
for k, (tr_idx, va_idx) in enumerate(folds):
    tr, va = dev.iloc[tr_idx], dev.iloc[va_idx]
    es_units = sorted(tr.unit.unique())
    np.random.default_rng(SEED + k).shuffle(es_units)
    es_units = set(es_units[:max(1, round(len(es_units) * DL["es_frac"]))])
    preds, F, info, _ = fit_all(tr, tr[~tr.unit.isin(es_units)], tr[tr.unit.isin(es_units)], {"va": va})
    rows = test_like_mask(F["va"])
    y = F["va"].rul.to_numpy()[rows]
    for name, p in preds["va"].items():
        oof.append({"fold": k, "model": name, **scores(y, p[rows])})
    fit_info += [{"stage": f"cv fold {k}", "model": m, **v} for m, v in info.items()]
    print(f"[{SUBSET}] fold {k}: " + ", ".join(f"{m} {s['rmse_capped']:.2f}" for m, s in
          ((r["model"], r) for r in oof if r["fold"] == k and r["model"] in TREES + DEEP)) + f" ({time.time() - t0:.0f}s)")

cv = pd.DataFrame(oof)
base = cv[cv.model == "hgb"].set_index("fold").rmse_capped
rows_ = []
for name, g in cv.groupby("model", sort=False):
    delta = g.set_index("fold").rmse_capped - base
    rows_.append({"model": name, **g.drop(columns=["fold", "model"]).mean().to_dict(),
                  "fold_std_rmse_capped": float(g.rmse_capped.std()), "delta_vs_hgb": float(delta.mean()),
                  "folds_better_than_hgb": f"{int((delta < 0).sum())}/{len(delta)}"})
cv_table = pd.DataFrame(rows_).set_index("model").round(3)
PER_SEED = [m for m in cv_table.index if "[seed " in m]

# 2) official test set: trees on the same engines as the production model (v2); nets early-stop on the
#    validation engines (never used for training here), one prediction per test engine at its last cycle
preds, F, info, prep = fit_all(dev, dev, val_df, {"test": test_df})
fit_info += [{"stage": "test fit", "model": m, **v} for m, v in info.items()]
last = F["test"].groupby("unit").cycle.idxmax().to_numpy()
y_test = F["test"].rul.to_numpy()[last]
test_rows = []
for name, p in preds["test"].items():
    pt = p[last]
    test_rows.append({"model": name, "test_rmse_capped": float(np.sqrt(np.mean((pt - np.minimum(y_test, EVAL_CAP)) ** 2))),
                      "test_rmse": float(np.sqrt(np.mean((pt - y_test) ** 2))),
                      "test_nasa_score": float(nasa_score(y_test, pt).sum()),
                      "test_nasa_capped_mean": float(nasa_score(np.minimum(y_test, EVAL_CAP), pt).mean())})
test_table = pd.DataFrame(test_rows).set_index("model").round(3)

# how much of each net's score is luck of the random start: spread over its seeds, vs its seed ensemble
seed_rows = []
for kind in DEEP:
    single = [f"{kind} [seed {s}]" for s in DL["seeds"]]
    cvs, tests = cv_table.loc[single, "rmse_capped"], test_table.loc[single, "test_rmse_capped"]
    seed_rows.append({"net": kind, "cv_single_seed_mean": cvs.mean(), "cv_single_seed_std": cvs.std(),
                      "cv_single_seed_range": cvs.max() - cvs.min(), "cv_seed_ensemble": cv_table.loc[kind, "rmse_capped"],
                      "cv_single_seed_delta_vs_hgb_mean": cv_table.loc[single, "delta_vs_hgb"].mean(),
                      "test_single_seed_mean": tests.mean(), "test_single_seed_range": tests.max() - tests.min(),
                      "test_seed_ensemble": test_table.loc[kind, "test_rmse_capped"]})
seed_table = pd.DataFrame(seed_rows).set_index("net").round(3)

# 3) verdict, same spirit as the promotion gate: a net replaces boosting only if it is clearly better
#    (capped CV RMSE lower by more than the 0.5-cycle margin, in at least 4 of 5 paired folds). Candidates are
#    the deployable models: the seed-ensembled nets and the blends (single-seed runs are diagnostics only)
cand = cv_table.drop(index=TREES + PER_SEED)
best = cand.rmse_capped.idxmin()
wins = int(cand.loc[best, "folds_better_than_hgb"].split("/")[0])
adopt = cand.loc[best, "delta_vs_hgb"] < -CFG["noninferiority_margin"] and wins >= 4
verdict = {"best_non_tree_model": best, "delta_vs_hgb_cycles": float(cand.loc[best, "delta_vs_hgb"]),
           "folds_better": cand.loc[best, "folds_better_than_hgb"], "adopt_as_challenger": bool(adopt),
           "decision": (f"{best} enters the promotion gate as a challenger" if adopt
                        else "keep gradient boosting: no deep model or blend is clearly better")}

display(cv_table.drop(index=PER_SEED))
display(test_table.drop(index=PER_SEED))
display(seed_table)
print(verdict)
DEEP_OUT = OUT / "deep"
DEEP_OUT.mkdir(parents=True, exist_ok=True)
cv_table.to_csv(DEEP_OUT / "cv_table.csv")
cv.to_csv(DEEP_OUT / "cv_per_fold.csv", index=False)
test_table.to_csv(DEEP_OUT / "test_table.csv")
seed_table.to_csv(DEEP_OUT / "seed_spread.csv")
pd.DataFrame(fit_info).to_csv(DEEP_OUT / "training_info.csv", index=False)
save_json({"subset": SUBSET, "engines": int(dev.unit.nunique()), "folds": CFG["cv_folds"], "dl_config": DL,
           "sensors": prep["sensors"], "device": str(DEVICE), "verdict": verdict,
           "cv": cv_table.reset_index().to_dict(orient="records"),
           "test": test_table.reset_index().to_dict(orient="records"),
           "seed_spread": seed_table.reset_index().to_dict(orient="records")}, DEEP_OUT / "deep.json")
DEEP_RESULT = {"cv": cv_table, "test": test_table, "seeds": seed_table, "verdict": verdict,
               "info": pd.DataFrame(fit_info)}
"""


def deep_cells(pipeline_cells):
    # base pipeline cells up to the model definitions (no tracking, no training), then the deep-model code
    code_cells = []
    for kind, src, tags in pipeline_cells:
        if kind != "code" or 'ensure("mlflow")' in src or "class Tracker" in src:
            continue
        if "train" in tags:
            break
        code_cells.append(src)
    assert any("def make_rul_model" in s for s in code_cells), "model definitions missing from deep-model base"
    payload = json.dumps(code_cells + [DEEP_DEFS_CELL.strip("\n"), DEEP_RUN_CELL.strip("\n")], indent=0)
    assert "'''" not in payload
    cells = [("markdown", r"""
# Deep sequence models vs gradient boosting

Do models built for temporal data beat our gradient-boosting champion? Three standard architectures for C-MAPSS are trained on **windows of the last 30 cycles of per-condition-normalised sensors** (past only, checked by a causality test):

| Model | Idea |
|---|---|
| **1D-CNN** | stacked 1D convolutions over time (the classic C-MAPSS conv net) |
| **GRU** | two-layer recurrent network, read out at the latest cycle |
| **TCN** | causal dilated convolutions with residual blocks (receptive field 61 cycles) |

**Fair comparison**
- Same cleaning, per-condition normalisation, RUL cap (125) and engine-level splits as the pipeline. The nets get the raw normalised sensors plus `cycle`; the trees keep their engineered features. That's the point of the comparison.
- **The same 5-fold GroupKFold by engine** as the ablation study, so every fold is a paired comparison. Nets early-stop on 15% of each fold's training engines; HistGB and XGBoost are refit in the same folds.
- **Official test set:** every model is trained on the same engines as the production model (v2). The nets early-stop on the validation engines, which nothing trains on. One prediction per test engine at its last cycle.
- **3 random starts per net (seeds 42, 1, 2).** A net's score depends on its initial weights and batch order, so one seed could win or lose by luck. Each net is trained 3 times in every fold and on the test fit. The net as deployed is the average of its 3 seeds (a seed ensemble), and the seed-spread table shows how far single seeds scatter around it.
- **Blends:** HistGB + each net, all three nets, and HistGB + all three. A different model family may help where the HistGB + XGBoost blend didn't.
- **Verdict** (same spirit as the promotion gate): a net or blend replaces boosting only if its capped CV RMSE is lower by more than the 0.5-cycle margin in at least 4 of 5 folds. Otherwise boosting stays: it's faster, explainable and lighter.

**Run on Kaggle**: attach the dataset, *Accelerator → GPU T4* (one GPU is used), then *Save Version → Save & Run All*. Expect roughly 2.5–3 hours for all four subsets (3 nets × 3 seeds × 6 fits each); to split it across sessions, shorten `SUBSETS`. Output: `/kaggle/working/deep_outputs.zip`. Generated from `scripts/build_notebook.py`; edit it there.
""".strip("\n"), [])]
    cells.append(("code", "import json\n\n# pipeline code cells + deep-model code, generated from scripts/build_notebook.py\n"
                  "DEEP_CELLS = json.loads(r'''" + payload + "''')\n"
                  "print(len(DEEP_CELLS), 'cells loaded')", []))
    cells.append(("code", r"""
import gc, shutil, time, traceback
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
WORK = Path("/kaggle/working") if Path("/kaggle/input").exists() else Path.cwd() / "outputs"
shutil.rmtree(WORK / "artifacts", ignore_errors=True)

deep = {}
for subset in SUBSETS:
    print(f"\n{'=' * 34} {subset} {'=' * 34}")
    ns = {"__name__": "__main__", "display": display}
    t0 = time.time()
    try:
        for i, src in enumerate(DEEP_CELLS):
            exec(compile(src.replace('SUBSET = "FD001"', f'SUBSET = "{subset}"'), f"<{subset} cell {i}>", "exec"), ns)
        deep[subset] = ns["DEEP_RESULT"]
    except Exception:
        traceback.print_exc()
    finally:
        plt.close("all")
        if "torch" in ns:
            ns["torch"].cuda.empty_cache()
        del ns
        gc.collect()
    print(f"{subset}: {'ok' if subset in deep else 'FAILED'} in {(time.time() - t0) / 60:.1f} min")
""".strip("\n"), ["train"]))
    cells.append(("markdown", r"""
## Cross-subset summary

`delta_vs_hgb` is the change in capped CV RMSE (cycles) against HistGradientBoosting, averaged over the 5 paired folds: negative = better than boosting. The test table is the official NASA test set (capped and true RMSE, NASA score).
""".strip("\n"), []))
    cells.append(("code", r"""
if deep:
    order = list(dict.fromkeys(m for r in deep.values() for m in r["cv"].index if "[seed " not in m))
    cv_rmse = pd.DataFrame({s: r["cv"]["rmse_capped"] for s, r in deep.items()}).reindex(order)
    cv_delta = pd.DataFrame({s: r["cv"]["delta_vs_hgb"] for s, r in deep.items()}).reindex(order)
    cv_wins = pd.DataFrame({s: r["cv"]["folds_better_than_hgb"] for s, r in deep.items()}).reindex(order)
    test_capped = pd.DataFrame({s: r["test"]["test_rmse_capped"] for s, r in deep.items()}).reindex(order)
    test_true = pd.DataFrame({s: r["test"]["test_rmse"] for s, r in deep.items()}).reindex(order)
    verdicts = pd.DataFrame({s: r["verdict"] for s, r in deep.items()})
    seeds = pd.concat({s: r["seeds"] for s, r in deep.items()})
    for title, t in [("capped RMSE (cycles), CV on training engines; nets = average of 3 seeds", cv_rmse),
                     ("change vs HistGB (cycles; - = better)", cv_delta.round(2)),
                     ("folds where the model beat HistGB", cv_wins),
                     ("official test set, capped RMSE", test_capped), ("official test set, true RMSE", test_true),
                     ("seed spread per net (single seeds vs their average)", seeds), ("verdict", verdicts)]:
        print(title)
        display(t)
    out = WORK / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    cv_rmse.to_csv(out / "deep_cv_rmse_capped.csv")
    cv_delta.to_csv(out / "deep_cv_delta_vs_hgb.csv")
    cv_wins.to_csv(out / "deep_cv_folds_better.csv")
    test_capped.to_csv(out / "deep_test_rmse_capped.csv")
    test_true.to_csv(out / "deep_test_rmse.csv")
    verdicts.to_csv(out / "deep_verdicts.csv")
    seeds.to_csv(out / "deep_seed_spread.csv")
    pd.concat({s: r["info"] for s, r in deep.items()}).to_csv(out / "deep_training_info.csv")
    zip_path = shutil.make_archive(str(WORK / "deep_outputs"), "zip", root_dir=out)
    print("outputs:", zip_path)
""".strip("\n"), ["train"]))
    return cells


def to_nb(cells):
    out = []
    for kind, src, tags in cells:
        lines = src.splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {"tags": tags} if tags else {}, "source": lines}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        out.append(cell)
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def run_all_cells(pipeline_cells):
    # every pipeline code cell except the pip install, embedded so both notebooks always match
    code_cells = [src for kind, src, _ in pipeline_cells if kind == "code" and 'ensure("mlflow")' not in src]
    payload = json.dumps(code_cells, indent=0)
    assert "'''" not in payload and sum('SUBSET = "FD001"' in s for s in code_cells) == 1
    assert sum("SEED = 42\n" in s for s in code_cells) == 1 and sum('RUN_TAG = ""' in s for s in code_cells) == 1
    cells = []
    cells.append(("markdown", r"""
# Run the pipeline on all four C-MAPSS subsets

Runs the full pipeline from `pipeline_FD001.ipynb` (ingest → features → CV model selection → retraining and promotion → evaluation → triage agent → export) on FD001, FD002, FD003 and FD004, one after another, without manual steps.

- Each subset runs in a **fresh namespace**, the same as restarting the kernel and changing `SUBSET` by hand.
- If one subset fails, its error is recorded and the **next subset still runs**.
- `CLEAN_START = True` wipes earlier artifacts and the tracking store first, so outputs only contain this run.
- At the end: a **cross-subset summary table**, the **monitoring dashboard** (`dashboard.html`, one offline page, inside the zip) and **one zip** with every subset's artifacts plus one MLflow store (one experiment per subset): `/kaggle/working/all_subsets_outputs.zip`.

**Run on Kaggle**: attach the dataset, *Accelerator → GPU T4 x2*, *Internet → On*, then *Run all*. To run only some subsets, edit `SUBSETS`. The pipeline code is embedded below and generated from the same source as `pipeline_FD001.ipynb` (`scripts/build_notebook.py`); edit it there, not here.
""".strip("\n"), []))
    cells.append(("code", r"""
# MLflow is not preinstalled on Kaggle. If the install fails, the pipeline falls back to JSON tracking.
import importlib.util, subprocess, sys

def ensure(module, pip_name=None):
    if importlib.util.find_spec(module) is None:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name or module],
                           capture_output=True, text=True)
        print(f"pip install {pip_name or module}: {'ok' if r.returncode == 0 else 'FAILED'}")

ensure("mlflow")
""".strip("\n"), []))
    cells.append(("code", "import json\n\n# pipeline code cells, generated from scripts/build_notebook.py\n"
                  "PIPELINE_CELLS = json.loads(r'''" + payload + "''')\n"
                  "# monitoring dashboard builder, embedded from scripts/build_dashboard.py\n"
                  "DASHBOARD_SRC = " + repr(DASHBOARD_SRC) + "\n"
                  "print(len(PIPELINE_CELLS), 'pipeline cells loaded')", []))
    cells.append(("code", r"""
import gc, shutil, time, traceback
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
# Seed 42 is the primary run (outputs in artifacts/<SUBSET>, as before). Each extra seed changes the engine splits,
# CV folds and model randomness; its outputs go to artifacts/<SUBSET>_seed<N>. Use [42] for a quick single run.
SEEDS = [42, 1, 2, 3, 4]
CLEAN_START = True

WORK = Path("/kaggle/working") if Path("/kaggle/input").exists() else Path.cwd() / "outputs"
if CLEAN_START:
    for d in ["artifacts", "tracking"] + [f"tracking_seed{s}" for s in SEEDS]:
        shutil.rmtree(WORK / d, ignore_errors=True)

results = {}  # (subset, seed) -> result
for seed in SEEDS:
    tag = "" if seed == 42 else f"_seed{seed}"
    for subset in SUBSETS:
        print(f"\n{'=' * 30} {subset} · seed {seed} {'=' * 30}")
        ns = {"__name__": "__main__", "display": display, "SKIP_ZIP": True}
        t0 = time.time()
        try:
            for i, src in enumerate(PIPELINE_CELLS):
                code = (src.replace('SUBSET = "FD001"', f'SUBSET = "{subset}"')
                           .replace("SEED = 42\n", f"SEED = {seed}\n")
                           .replace('RUN_TAG = ""', f'RUN_TAG = "{tag}"'))
                exec(compile(code, f"<{subset} seed {seed} cell {i}>", "exec"), ns)
            results[(subset, seed)] = {"status": "ok", "minutes": round((time.time() - t0) / 60, 1),
                                       "monitoring": ns["monitoring"]}
        except Exception:
            traceback.print_exc()
            results[(subset, seed)] = {"status": "failed", "minutes": round((time.time() - t0) / 60, 1),
                                       "error": traceback.format_exc(limit=5)}
        finally:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass
            plt.close("all")
            del ns
            gc.collect()
        print(f"{subset} seed {seed}: {results[(subset, seed)]['status']} in {results[(subset, seed)]['minutes']} min")
""".strip("\n"), ["train"]))
    cells.append(("markdown", r"""
## Cross-subset summary

`test_rmse_capped` is the error against min(true RUL, 125), the part the model can reduce; `test_rmse` is against the true RUL. FD002 / FD004 have ~25% of test engines above the cap, which drives their true RMSE and NASA score (see each subset's `test_error_by_band` in `monitoring.json`).

The first table is the primary run (seed 42). The robustness table repeats the key numbers across all seeds: mean, standard deviation, and the worst seed, so a result that only holds for one lucky split shows up here.
""".strip("\n"), []))
    cells.append(("code", r"""
def engines_above_cap(m):
    cap = m["config"]["rul_cap"]
    return int(sum(v["engines"] for k, v in m["test_error_by_band"].items() if float(k.strip("([").split(",")[0]) >= cap))


def summary_row(r):
    if r["status"] != "ok":
        return {"status": "FAILED", "minutes": r["minutes"]}
    m = r["monitoring"]
    p, a = m["production_metrics"], m["agent_summary"]
    return {
        "status": "ok", "model": m["champion_model"], "production_version": m["production_version"],
        "cv_rmse_capped": round(m["cv_results"][m["champion_model"]]["rmse_capped"], 2),
        "test_rmse_capped": round(p["test_rmse_capped"], 2), "test_rmse": round(p["test_rmse"], 2),
        "test_nasa_score": round(p["test_nasa_score"], 1), "test_interval_coverage": round(p["test_interval_coverage"], 3),
        "test_engines": a["engines"], "test_engines_above_cap": engines_above_cap(m),
        "urgent_high_or_critical": f"{a['urgent_high_or_critical']}/{a['urgent_engines']}",
        "urgent_caught": f"{a['urgent_caught']}/{a['urgent_engines']}", "urgent_missed": a["urgent_missed"],
        "exact_severity_pct": round(a["exact_severity_match_pct"], 1), "review_pct": round(a["review_rate_pct"], 1),
        "false_alarms": a["false_alarms_true_rul_gt_low"], "work_orders": a["work_orders"],
        "promoted_v2": m["promotion"].get("promoted"),
        "v2_better_in_bootstrap_pct": round(100 * m["promotion"].get("bootstrap_prob_new_better", float("nan")), 1),
        "val_engines": m["fleet_replay"]["engines"],
        "ours_unplanned_failures": m["policy_comparison"]["predictive agent (ours)"]["unplanned_failures"],
        "ours_life_used_pct": round(m["policy_comparison"]["predictive agent (ours)"]["life_used_pct"], 1),
        "fixed_safe_life_used_pct": round(next(v["life_used_pct"] for k, v in m["policy_comparison"].items()
                                               if "0% failures" in k), 1),
        "median_rul_at_first_order": m["fleet_replay"]["median_rul_at_first_order"],
        "fault_injections_detected": f"{sum(f['detected'] for f in m['fault_injection'])}/{len(m['fault_injection'])}",
        "minutes": r["minutes"],
    }


primary = SEEDS[0]
summary = pd.DataFrame({s: summary_row(results[(s, primary)]) for s in SUBSETS if (s, primary) in results})
print(f"primary run (seed {primary})")
display(summary)

for (subset, seed), r in results.items():
    if r["status"] != "ok":
        print(f"\n{subset} seed {seed} failed:\n{r['error']}")

# robustness across seeds: numeric headline metrics per (subset, seed), then mean / std / worst per subset
def seed_metrics(r):
    m = r["monitoring"]
    p, a, pol = m["production_metrics"], m["agent_summary"], m["policy_comparison"]
    ours = pol["predictive agent (ours)"]
    safe = next(v for k, v in pol.items() if "0% failures" in k)
    return {"test_rmse_capped": p["test_rmse_capped"], "test_rmse": p["test_rmse"],
            "test_nasa_score": p["test_nasa_score"], "test_interval_coverage": p["test_interval_coverage"],
            "urgent_high_or_critical_pct": 100 * a["urgent_high_or_critical"] / a["urgent_engines"],
            "urgent_missed": a["urgent_missed"], "false_alarms": a["false_alarms_true_rul_gt_low"],
            "review_pct": a["review_rate_pct"], "promoted_v2": float(bool(m["promotion"].get("promoted"))),
            "ours_unplanned_failures": ours["unplanned_failures"], "ours_life_used_pct": ours["life_used_pct"],
            "fixed_safe_life_used_pct": safe["life_used_pct"]}


per_seed = pd.DataFrame([{"subset": s, "seed": sd, **seed_metrics(r)} for (s, sd), r in results.items()
                         if r["status"] == "ok"])
robust = None
if len(per_seed) and per_seed.seed.nunique() > 1:
    worst = {"test_rmse_capped": "max", "test_rmse": "max", "test_nasa_score": "max", "test_interval_coverage": "min",
             "urgent_high_or_critical_pct": "min", "urgent_missed": "max", "false_alarms": "max", "review_pct": "max",
             "promoted_v2": "min", "ours_unplanned_failures": "max", "ours_life_used_pct": "min",
             "fixed_safe_life_used_pct": "max"}
    g = per_seed.groupby("subset")
    robust = pd.concat({"mean": g[list(worst)].mean(), "std": g[list(worst)].std(),
                        "worst": g.agg(worst)}, axis=1).T.round(2)
    print(f"robustness across {per_seed.seed.nunique()} seeds {sorted(per_seed.seed.unique())}")
    display(robust)

(WORK / "artifacts").mkdir(parents=True, exist_ok=True)
summary.to_csv(WORK / "artifacts" / "summary_all_subsets.csv")
(WORK / "artifacts" / "summary_all_subsets.json").write_text(summary.to_json(indent=2))
per_seed.to_csv(WORK / "artifacts" / "summary_per_seed.csv", index=False)
if robust is not None:
    robust.to_csv(WORK / "artifacts" / "summary_robustness.csv")

bundle_dir = WORK / "all_subsets_outputs"
shutil.rmtree(bundle_dir, ignore_errors=True)
# extra seeds are for robustness numbers only: keep their results, drop their model files to keep the zip small
skip_models = lambda d, names: [n for n in names if "_seed" in str(d) and (n.endswith(".joblib") or n == "models")]
shutil.copytree(WORK / "artifacts", bundle_dir / "artifacts", ignore=skip_models)
if (WORK / "tracking").exists():
    shutil.copytree(WORK / "tracking", bundle_dir / "tracking")  # primary seed's MLflow store only
# monitoring dashboard: one offline HTML page built from the primary seed's artifacts
dash_ns = {"__name__": "build_dashboard"}
exec(compile(DASHBOARD_SRC, "build_dashboard.py", "exec"), dash_ns)
print("dashboard:", dash_ns["build"](WORK, bundle_dir / "dashboard.html"))
zip_path = shutil.make_archive(str(bundle_dir), "zip", root_dir=bundle_dir)
shutil.rmtree(bundle_dir)
print("outputs:", zip_path)
""".strip("\n"), ["train"]))
    return cells


if __name__ == "__main__":
    nb_dir = Path(__file__).resolve().parents[1] / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    for name, cells in [("pipeline_FD001.ipynb", CELLS), ("run_all_subsets.ipynb", run_all_cells(CELLS)),
                        ("ablation.ipynb", ablation_cells(CELLS)), ("deep_models.ipynb", deep_cells(CELLS))]:
        (nb_dir / name).write_text(json.dumps(to_nb(cells), indent=1), encoding="utf-8")
        print(f"wrote {nb_dir / name} ({len(cells)} cells)")
