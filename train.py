"""
CrimeWatch Analytics — Enhanced Training Script
================================================
Run once:  python3 train.py

Produces model_files/ with:
  xgb_model.pkl    — trained XGBoost regressor
  encoders.pkl     — LabelEncoders for state / district / crime
  metrics.pkl      — MAE, RMSE, R², CV scores
  analytics.json   — pre-computed dashboard data

Improvements over original:
  • 5-fold cross-validation → CV MAE/RMSE added to metrics.pkl
  • Feature-importance printed and stored in analytics.json
  • Per-crime-type MAE breakdown saved for the /about page
  • YoY delta computed and stored (no repeated computation in app.py)
  • districts_by_state list is guaranteed sorted + deduped
  • Verbose progress with timestamps
  • Graceful CSV-not-found error message
"""

import sys
import time
import json
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor

# ── CONFIG ────────────────────────────────────────────────────
np.random.seed(42)
BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model_files")
CSV_PATH  = os.path.join(BASE, "crime.csv")

XGB_PARAMS = dict(
    n_estimators=200,       # increased for better accuracy
    max_depth=6,
    learning_rate=0.08,     # slightly lower LR with more trees
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,     # reduces overfitting
    gamma=0.1,              # reduces overfitting
    random_state=42,
    n_jobs=1,               # Docker-safe (no parallel fork issues)
    tree_method="hist",     # fastest CPU method
    verbosity=0,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── DATA LOADING ──────────────────────────────────────────────
log("train.py started")
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    print(f"ERROR: crime.csv not found at {CSV_PATH}", file=sys.stderr)
    sys.exit(1)

log("Loading crime.csv …")
df = pd.read_csv(CSV_PATH)
df.fillna(0, inplace=True)
df["STATE/UT"] = df["STATE/UT"].str.upper().str.strip()
df["DISTRICT"] = df["DISTRICT"].str.upper().str.strip()
df = df[df["DISTRICT"] != "TOTAL"].copy()
log(f"Loaded {len(df):,} rows | {df['STATE/UT'].nunique()} states | "
    f"{df['DISTRICT'].nunique()} districts")

CRIME_COLS = [c for c in df.columns
              if c not in ["STATE/UT", "DISTRICT", "YEAR", "TOTAL IPC CRIMES"]]
log(f"Crime types detected: {len(CRIME_COLS)}")


# ── MELT TO LONG FORMAT ───────────────────────────────────────
log("Melting to long format …")
long_df = df.melt(
    id_vars=["STATE/UT", "DISTRICT", "YEAR"],
    value_vars=CRIME_COLS,
    var_name="CRIME_TYPE",
    value_name="COUNT",
)
log(f"Long format: {len(long_df):,} rows")


# ── LABEL ENCODING ────────────────────────────────────────────
log("Encoding labels …")
state_enc    = LabelEncoder()
district_enc = LabelEncoder()
crime_enc    = LabelEncoder()

long_df["STATE_ENC"]    = state_enc.fit_transform(long_df["STATE/UT"])
long_df["DISTRICT_ENC"] = district_enc.fit_transform(long_df["DISTRICT"])
long_df["CRIME_ENC"]    = crime_enc.fit_transform(long_df["CRIME_TYPE"])

X = long_df[["YEAR", "STATE_ENC", "DISTRICT_ENC", "CRIME_ENC"]]
y = long_df["COUNT"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
log(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")


# ── TRAINING ──────────────────────────────────────────────────
log("Training XGBoost …")
t0 = time.time()
model = XGBRegressor(**XGB_PARAMS)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
log(f"Training complete in {time.time() - t0:.1f}s")


# ── EVALUATION ────────────────────────────────────────────────
log("Evaluating …")
preds = model.predict(X_test)
mae   = mean_absolute_error(y_test, preds)
r2    = r2_score(y_test, preds)
rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
log(f"MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")

# 5-fold cross-validation (on a 20 % sample to keep it fast)
log("Running 5-fold cross-validation (sampled) …")
sample_mask = np.random.choice(len(X), size=min(50_000, len(X)), replace=False)
X_sample    = X.iloc[sample_mask]
y_sample    = y.iloc[sample_mask]
kf          = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores   = cross_val_score(
    XGBRegressor(**XGB_PARAMS), X_sample, y_sample,
    cv=kf, scoring="neg_mean_absolute_error", n_jobs=1,
)
cv_mae = float(-cv_scores.mean())
cv_std = float(cv_scores.std())
log(f"CV MAE: {cv_mae:.2f} ± {cv_std:.2f}")

# Per-crime MAE breakdown (test set)
log("Computing per-crime MAE …")
test_df          = long_df.iloc[y_test.index].copy()
test_df["PRED"]  = preds
crime_mae = (
    test_df.groupby("CRIME_TYPE")
    .apply(lambda g: round(mean_absolute_error(g["COUNT"], g["PRED"]), 2))
    .sort_values(ascending=False)
    .head(10)
    .to_dict()
)

# Feature importances
feat_names   = ["Year", "State", "District", "Crime Type"]
importances  = {
    feat_names[i]: round(float(v), 4)
    for i, v in enumerate(model.feature_importances_)
}
log(f"Feature importances: {importances}")


# ── ANALYTICS JSON ────────────────────────────────────────────
log("Building analytics.json …")

year_trend  = df.groupby("YEAR")["TOTAL IPC CRIMES"].sum().reset_index()
state_total = df.groupby("STATE/UT")["TOTAL IPC CRIMES"].sum().nlargest(10).reset_index()
crime_totals = {c: int(df[c].sum()) for c in CRIME_COLS}
top_crimes   = sorted(crime_totals.items(), key=lambda x: x[1], reverse=True)[:10]

# YoY deltas
years_list  = year_trend["YEAR"].tolist()
totals_list = [int(v) for v in year_trend["TOTAL IPC CRIMES"].tolist()]
yoy_deltas  = [0] + [totals_list[i] - totals_list[i - 1]
                     for i in range(1, len(totals_list))]

# districts_by_state — sorted + deduped
districts_by_state = (
    df.groupby("STATE/UT")["DISTRICT"]
    .apply(lambda x: sorted(set(x.tolist())))
    .to_dict()
)

# Top 10 district hotspots
hotspots = (
    df.groupby(["STATE/UT", "DISTRICT"])["TOTAL IPC CRIMES"]
    .sum()
    .nlargest(10)
    .reset_index()
    .rename(columns={"STATE/UT": "state", "DISTRICT": "district",
                     "TOTAL IPC CRIMES": "total"})
)

analytics = {
    "year_trend": {
        "years":  years_list,
        "totals": totals_list,
        "yoy_deltas": yoy_deltas,
    },
    "state_data": {
        "states": state_total["STATE/UT"].tolist(),
        "totals": [int(v) for v in state_total["TOTAL IPC CRIMES"].tolist()],
    },
    "top_crimes":        [{"crime": c, "count": n} for c, n in top_crimes],
    "crime_cols":        CRIME_COLS,
    "states":            sorted(df["STATE/UT"].unique().tolist()),
    "years":             sorted(df["YEAR"].unique().tolist()),
    "districts_by_state": districts_by_state,
    "hotspots":          hotspots.to_dict("records"),
    "feature_importances": importances,
    "per_crime_mae":     crime_mae,
}


# ── SAVE ─────────────────────────────────────────────────────
log("Saving artefacts …")

joblib.dump(model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
joblib.dump(
    {"state": state_enc, "district": district_enc, "crime": crime_enc},
    os.path.join(MODEL_DIR, "encoders.pkl"),
)
joblib.dump(
    {
        "mae":    float(mae),
        "rmse":   float(rmse),
        "r2":     float(r2),
        "cv_mae": cv_mae,
        "cv_std": cv_std,
    },
    os.path.join(MODEL_DIR, "metrics.pkl"),
)

with open(os.path.join(MODEL_DIR, "analytics.json"), "w") as fh:
    json.dump(analytics, fh, indent=2)

log("All files saved to model_files/ — training complete!")
log(f"Summary  →  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}  "
    f"CV_MAE={cv_mae:.2f}±{cv_std:.2f}")
