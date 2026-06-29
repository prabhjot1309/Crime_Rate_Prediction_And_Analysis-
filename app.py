"""
CrimeWatch Analytics — Production-Safe Enhanced Flask App
=========================================================
Fixes applied vs original:
  1. Auto-detects BOTH model layouts:
       Layout A (new):  model_files/xgb_model.pkl + encoders.pkl + metrics.pkl + analytics.json
       Layout B (old):  model (1).pkl + crime_encoder (1).pkl + district_encoder (1).pkl (root)
  2. Wraps ALL asset loading in try/except with clear console errors
  3. /predict — full validation, confidence interval, yoy%, is_forecast flag
  4. /get_districts — optional ?include_stats=1 for state crime summary
  5. /predictions — accepts ?state=&district=&crime=&year= deep-link pre-fill
  6. /reports — top-20 states, YoY deltas, district hotspots, per-crime totals
  7. /data — district search, multi-col sort, dynamic per_page, CSV export link
  8. /data/export — NEW: filtered CSV download
  9. /about — feature importances, avg crimes/year, num districts
 10. /download_pdf — historical table, confidence row, hotspots table
 11. /api/state_summary, /api/hotspots, /api/compare, /api/trend — NEW JSON APIs
 12. /health — NEW: Docker/k8s liveness probe
 13. Global 404 / 405 / 500 error handlers
"""

import io
import json
import logging
import os
import traceback
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_file

# ── PDF ──────────────────────────────────────────────────────
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (HRFlowable, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
)
logger = logging.getLogger("crimewatch")

np.random.seed(42)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model_files")


# ══════════════════════════════════════════════════════════════
#  ASSET LOADING  (supports old layout AND new model_files/ layout)
# ══════════════════════════════════════════════════════════════

def _try_load_new_layout():
    """Load from model_files/ directory (new layout after train.py rewrite)."""
    model    = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    # encoders dict must have keys: 'state', 'district', 'crime'
    metrics  = joblib.load(os.path.join(MODEL_DIR, "metrics.pkl"))
    with open(os.path.join(MODEL_DIR, "analytics.json")) as fh:
        analytics = json.load(fh)
    return model, encoders, metrics, analytics


def _try_load_old_layout():
    """
    Load from root directory (original repo layout).
    Old files: 'model (1).pkl', 'crime_encoder (1).pkl', 'district_encoder (1).pkl'
    The old app used two separate encoders (crime + district) and no unified dict.
    We normalise them into the same dict structure here.
    """
    from sklearn.preprocessing import LabelEncoder

    # Try both naming variants
    model_candidates = ["model (1).pkl", "model.pkl", "xgb_model.pkl"]
    crime_enc_candidates = ["crime_encoder (1).pkl", "crime_encoder.pkl"]
    dist_enc_candidates  = ["district_encoder (1).pkl", "district_encoder.pkl"]

    model_path = next(
        (os.path.join(BASE, f) for f in model_candidates if os.path.exists(os.path.join(BASE, f))),
        None,
    )
    crime_enc_path = next(
        (os.path.join(BASE, f) for f in crime_enc_candidates if os.path.exists(os.path.join(BASE, f))),
        None,
    )
    dist_enc_path = next(
        (os.path.join(BASE, f) for f in dist_enc_candidates if os.path.exists(os.path.join(BASE, f))),
        None,
    )

    if not model_path:
        raise FileNotFoundError("No model pkl found in root directory")

    model       = joblib.load(model_path)
    crime_enc   = joblib.load(crime_enc_path)  if crime_enc_path  else LabelEncoder()
    dist_enc    = joblib.load(dist_enc_path)   if dist_enc_path   else LabelEncoder()

    # The old app had no state encoder stored separately — reconstruct from CSV
    df_tmp = pd.read_csv(os.path.join(BASE, "crime.csv"))
    df_tmp["STATE/UT"] = df_tmp["STATE/UT"].str.upper().str.strip()
    state_enc = LabelEncoder().fit(df_tmp["STATE/UT"].unique())

    encoders = {"state": state_enc, "district": dist_enc, "crime": crime_enc}

    # Stub metrics (old layout had none saved)
    metrics = {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "cv_mae": 0.0, "cv_std": 0.0}

    # Build analytics on-the-fly
    df_tmp = df_tmp[df_tmp["DISTRICT"] != "TOTAL"].copy()
    df_tmp["DISTRICT"] = df_tmp["DISTRICT"].str.upper().str.strip()
    df_tmp.fillna(0, inplace=True)
    crime_cols = [c for c in df_tmp.columns
                  if c not in ["STATE/UT", "DISTRICT", "YEAR", "TOTAL IPC CRIMES"]]

    year_trend  = df_tmp.groupby("YEAR")["TOTAL IPC CRIMES"].sum().reset_index()
    state_total = df_tmp.groupby("STATE/UT")["TOTAL IPC CRIMES"].sum().nlargest(10).reset_index()
    crime_totals = {c: int(df_tmp[c].sum()) for c in crime_cols}
    top_crimes   = sorted(crime_totals.items(), key=lambda x: x[1], reverse=True)[:10]
    districts_by_state = (
        df_tmp.groupby("STATE/UT")["DISTRICT"]
        .apply(lambda x: sorted(set(x.tolist())))
        .to_dict()
    )

    analytics = {
        "year_trend": {
            "years":  year_trend["YEAR"].tolist(),
            "totals": [int(v) for v in year_trend["TOTAL IPC CRIMES"].tolist()],
        },
        "state_data": {
            "states": state_total["STATE/UT"].tolist(),
            "totals": [int(v) for v in state_total["TOTAL IPC CRIMES"].tolist()],
        },
        "top_crimes":        [{"crime": c, "count": n} for c, n in top_crimes],
        "crime_cols":        crime_cols,
        "states":            sorted(df_tmp["STATE/UT"].unique().tolist()),
        "years":             sorted(df_tmp["YEAR"].unique().tolist()),
        "districts_by_state": districts_by_state,
        "feature_importances": {},
        "per_crime_mae": {},
    }
    return model, encoders, metrics, analytics


def _load_assets():
    """Try new layout first, fall back to old layout, crash clearly if both fail."""
    # 1. Try new layout
    if os.path.isdir(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "xgb_model.pkl")):
        try:
            result = _try_load_new_layout()
            logger.info("Loaded assets from model_files/ (new layout)")
            return result
        except Exception as e:
            logger.warning(f"New layout load failed: {e} — trying old layout …")

    # 2. Fall back to old layout
    try:
        result = _try_load_old_layout()
        logger.info("Loaded assets from root directory (old layout)")
        return result
    except Exception as e:
        logger.critical(f"FATAL: could not load any model assets — {e}")
        raise


# Load once at startup
model, encoders, metrics, analytics = _load_assets()

# Load DataFrame
_csv_path = os.path.join(BASE, "crime.csv")
df = pd.read_csv(_csv_path)
df.fillna(0, inplace=True)
df["STATE/UT"] = df["STATE/UT"].str.upper().str.strip()
df["DISTRICT"] = df["DISTRICT"].str.upper().str.strip()
df = df[df["DISTRICT"] != "TOTAL"].copy()

CRIME_COLS  = [c for c in df.columns
               if c not in ["STATE/UT", "DISTRICT", "YEAR", "TOTAL IPC CRIMES"]]
STATES      = analytics["states"]
DISTRICTS   = analytics["districts_by_state"]
YEARS       = analytics["years"]
FULL_YEARS  = list(range(2001, 2031))

# Pre-compute fast lookup tables
_state_crime_totals    = df.groupby("STATE/UT")[CRIME_COLS].sum()
_district_crime_totals = df.groupby(["STATE/UT", "DISTRICT"])[CRIME_COLS].sum()

logger.info(f"DataFrame: {len(df):,} rows | {df['STATE/UT'].nunique()} states | "
            f"{df['DISTRICT'].nunique()} districts | {len(CRIME_COLS)} crime types")


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════

def safe_encode(encoder, value):
    """Return encoded integer or None if value is unseen."""
    try:
        if value in encoder.classes_:
            return int(encoder.transform([value])[0])
    except Exception:
        pass
    return None


def predict_count(year: int, state: str, district: str, crime_type: str):
    """
    Return (predicted_count, low_ci, high_ci) or (None, None, None).
    Confidence interval is ±1.65 × MAE (approx 90 % CI).
    """
    s = safe_encode(encoders["state"],    state)
    d = safe_encode(encoders["district"], district)
    c = safe_encode(encoders["crime"],    crime_type)

    if any(v is None for v in [s, d, c]):
        return None, None, None

    try:
        raw = float(model.predict([[year, s, d, c]])[0])
    except Exception as e:
        logger.error(f"model.predict failed: {e}")
        return None, None, None

    val  = max(0.0, raw)
    mae  = metrics.get("mae", 0.0) or 0.0
    half = mae * 1.65
    return round(val, 1), round(max(0.0, val - half), 1), round(val + half, 1)


def forecast_series(state: str, district: str, crime_type: str,
                    from_year: int = 2014, to_year: int = 2030) -> dict:
    """Return {year: predicted_count} for the requested range."""
    s = safe_encode(encoders["state"],    state)
    d = safe_encode(encoders["district"], district)
    c = safe_encode(encoders["crime"],    crime_type)
    if any(v is None for v in [s, d, c]):
        return {}
    out = {}
    for yr in range(from_year, to_year + 1):
        try:
            val = float(model.predict([[yr, s, d, c]])[0])
            out[yr] = round(max(0.0, val), 1)
        except Exception:
            out[yr] = 0.0
    return out


def risk_label(count):
    if count is None:       return "Unknown",   "gray"
    if count == 0:          return "None",       "green"
    if count <= 35:         return "Low",        "green"
    if count <= 65:         return "Medium",     "yellow"
    if count <= 150:        return "High",       "orange"
    return "Very High", "red"


def pct_change(old, new):
    if old is None or old == 0:
        return None
    return round((new - old) / old * 100, 1)


def generate_ai_summary(state, district, crime_type, year, count, forecast,
                         low=None, high=None):
    label, _ = risk_label(count)
    trend_vals = list(forecast.values())

    direction = "stable"
    if len(trend_vals) >= 3:
        direction = "rising" if trend_vals[-1] > trend_vals[0] else "declining"

    yoy = pct_change(trend_vals[-2], trend_vals[-1]) if len(trend_vals) >= 2 else None

    lines = [
        f"📍 District: {district}, {state}",
        f"📅 Year: {year} | Crime: {crime_type}",
        "",
        f"🔍 Risk Level: {label}",
        f"📊 Predicted Cases: {count if count is not None else 'N/A'}",
    ]
    if low is not None and high is not None:
        lines.append(f"   Confidence Range (90%): {low} – {high} cases")

    lines += ["", f"📈 Forecast Trend (2014–2030): {direction.upper()}"]

    if yoy is not None:
        arrow = "↑" if yoy > 0 else "↓"
        lines.append(f"   Year-on-Year Change (latest): {arrow} {abs(yoy)}%")

    if direction == "rising":
        lines.append("⚠️  Increasing trend — enhanced patrolling may help.")
    elif direction == "declining":
        lines.append("✅ Declining trend — current measures appear effective.")
    else:
        lines.append("➡️  Stable trend — consistent enforcement recommended.")

    if crime_type in df.columns and state in _state_crime_totals.index:
        top3 = _state_crime_totals[crime_type].nlargest(3).index.tolist()
        if state in top3:
            lines.append(f"🚨 {state} is among the top 3 states for {crime_type}.")

        key = (state, district)
        if key in _district_crime_totals.index:
            dist_tot  = _district_crime_totals.loc[key, crime_type]
            state_tot = _state_crime_totals.loc[state, crime_type]
            if state_tot > 0:
                share = round(dist_tot / state_tot * 100, 1)
                lines.append(
                    f"📌 {district} contributes ~{share}% of {state}'s {crime_type} cases (2001-2013)."
                )
    return "\n".join(lines)


def _validate_predict_inputs(data: dict):
    """Parse + validate predict payload. Returns (year, state, district, crime_type)."""
    try:
        year = int(data.get("year", 2013))
    except (TypeError, ValueError):
        raise ValueError("'year' must be an integer.")

    state      = str(data.get("state",      "")).upper().strip()
    district   = str(data.get("district",   "")).upper().strip()
    crime_type = str(data.get("crime_type", "")).strip()

    if not state:
        raise ValueError("'state' is required.")
    if not district:
        raise ValueError("'district' is required.")
    if not crime_type:
        raise ValueError("'crime_type' is required.")
    if year < 2001 or year > 2030:
        raise ValueError(f"Year {year} is outside the supported range 2001–2030.")
    if state not in encoders["state"].classes_:
        raise ValueError(f"Unknown state: '{state}'. Please check spelling.")
    if district not in encoders["district"].classes_:
        raise ValueError(f"Unknown district: '{district}'. Please check spelling.")
    if crime_type not in encoders["crime"].classes_:
        raise ValueError(f"Unknown crime type: '{crime_type}'.")

    return year, state, district, crime_type


# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    total_crimes   = int(df["TOTAL IPC CRIMES"].sum())
    district_count = int(df["DISTRICT"].nunique())
    avg_per_dist   = round(total_crimes / district_count) if district_count else 0
    year_min       = int(df["YEAR"].min())
    year_max       = int(df["YEAR"].max())
    crime_totals   = {c: int(df[c].sum()) for c in CRIME_COLS}
    top_crime      = max(crime_totals, key=crime_totals.get) if crime_totals else ""

    return render_template(
        "index.html",
        states=STATES,
        years=FULL_YEARS,
        crime_cols=CRIME_COLS,
        year_trend=json.dumps(analytics["year_trend"]),
        top_crimes=json.dumps(analytics["top_crimes"]),
        state_data=json.dumps(analytics["state_data"]),
        total_crimes=f"{total_crimes:,}",
        district_count=f"{district_count:,}",
        avg_per_dist=f"{avg_per_dist:,}",
        year_range=f"{year_min}–{year_max}",
        top_crime=top_crime,
        crime_type_count=len(CRIME_COLS),
        model_r2=round(metrics.get("r2", 0) * 100, 1),
        model_mae=round(metrics.get("mae", 0), 1),
    )


# ─────────────────────────────────────────────────────────────
@app.route("/get_districts")
def get_districts():
    state         = request.args.get("state", "").upper().strip()
    include_stats = request.args.get("include_stats", "0") == "1"
    districts     = DISTRICTS.get(state, [])

    if not include_stats:
        return jsonify(districts)

    state_df   = df[df["STATE/UT"] == state]
    dist_tots  = (
        state_df.groupby("DISTRICT")["TOTAL IPC CRIMES"]
        .sum().sort_values(ascending=False).head(5).to_dict()
    )
    top_cr = {}
    if state in _state_crime_totals.index:
        top_cr = (
            _state_crime_totals.loc[state]
            .sort_values(ascending=False).head(5).to_dict()
        )
    return jsonify({"districts": districts,
                    "top_districts_by_crime": dist_tots,
                    "top_crimes": top_cr})


# ─────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    # ── 1. Parse body ──────────────────────────────────────────
    try:
        data = request.get_json(force=True, silent=True)
        if data is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400
    except Exception:
        return jsonify({"error": "Could not parse request body as JSON."}), 400

    # ── 2. Validate inputs ────────────────────────────────────
    try:
        year, state, district, crime_type = _validate_predict_inputs(data)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # ── 3. Predict ────────────────────────────────────────────
    try:
        count, low, high = predict_count(year, state, district, crime_type)

        if count is None:
            return jsonify({"error": "Prediction failed — encoder mismatch."}), 500

        label, color = risk_label(count)
        from_yr      = max(year, 2014)
        forecast     = forecast_series(state, district, crime_type,
                                       from_year=from_yr, to_year=2030)

        # Historical actuals
        hist_sub   = df[(df["STATE/UT"] == state) & (df["DISTRICT"] == district)]
        historical = {}
        if crime_type in df.columns:
            for _, row in hist_sub.iterrows():
                historical[int(row["YEAR"])] = int(row[crime_type])

        hist_avg = round(float(np.mean(list(historical.values()))), 1) if historical else None
        fvals    = list(forecast.values())
        yoy_pct  = pct_change(fvals[-2], fvals[-1]) if len(fvals) >= 2 else None

        summary = generate_ai_summary(
            state, district, crime_type, year, count, forecast, low, high
        )

        return jsonify({
            "count":       count,
            "low":         low,
            "high":        high,
            "label":       label,
            "color":       color,
            "is_forecast": year > int(df["YEAR"].max()),
            "hist_avg":    hist_avg,
            "yoy_pct":     yoy_pct,
            "summary":     summary,
            "forecast":    {str(k): v for k, v in forecast.items()},
            "historical":  {str(k): v for k, v in historical.items()},
        })

    except Exception as e:
        logger.error(f"/predict error: {traceback.format_exc()}")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


# ─────────────────────────────────────────────────────────────
@app.route("/predictions")
def predictions():
    prefill = {
        "state":    request.args.get("state",    "").upper().strip(),
        "district": request.args.get("district", "").upper().strip(),
        "crime":    request.args.get("crime",    ""),
        "year":     request.args.get("year",     ""),
    }
    return render_template(
        "predictions.html",
        states=STATES,
        crime_cols=CRIME_COLS,
        years=FULL_YEARS,
        prefill=prefill,
    )


# ─────────────────────────────────────────────────────────────
@app.route("/reports")
def reports():
    state_summary = (
        df.groupby("STATE/UT")
        .agg(total_crimes=("TOTAL IPC CRIMES", "sum"),
             years_count=("YEAR", "nunique"),
             districts=("DISTRICT", "nunique"))
        .reset_index()
        .sort_values("total_crimes", ascending=False)
        .head(20)
    )
    state_summary["rank"] = range(1, len(state_summary) + 1)

    yt          = analytics["year_trend"]
    totals_list = yt["totals"]
    yoy_deltas  = [0] + [totals_list[i] - totals_list[i - 1]
                         for i in range(1, len(totals_list))]

    hotspots = (
        df.groupby(["STATE/UT", "DISTRICT"])["TOTAL IPC CRIMES"]
        .sum().nlargest(10).reset_index()
        .rename(columns={"STATE/UT": "state", "DISTRICT": "district",
                         "TOTAL IPC CRIMES": "total"})
    )

    crime_totals = sorted(
        [{"crime": c, "total": int(df[c].sum())} for c in CRIME_COLS],
        key=lambda x: x["total"], reverse=True,
    )

    return render_template(
        "reports.html",
        state_rows=state_summary.to_dict("records"),
        hotspot_rows=hotspots.to_dict("records"),
        crime_totals=json.dumps(crime_totals),
        yoy_years=json.dumps(yt["years"]),
        yoy_deltas=json.dumps(yoy_deltas),
        model_r2=round(metrics.get("r2", 0) * 100, 1),
        model_mae=round(metrics.get("mae", 0), 1),
        model_rmse=round(metrics.get("rmse", 0), 1),
        top_crimes=json.dumps(analytics["top_crimes"]),
        state_data=json.dumps(analytics["state_data"]),
        year_trend=json.dumps(analytics["year_trend"]),
    )


# ─────────────────────────────────────────────────────────────
@app.route("/data")
def data_page():
    page     = max(1, int(request.args.get("page", 1)))
    per_page = int(request.args.get("per_page", 50))
    if per_page not in (10, 25, 50, 100):
        per_page = 50

    state_filter    = request.args.get("state",          "").upper().strip()
    year_filter     = request.args.get("year",           "")
    district_search = request.args.get("district_search","").upper().strip()
    sort_by         = request.args.get("sort_by",        "TOTAL IPC CRIMES")
    sort_order      = request.args.get("order",          "desc")

    valid_sort = {"STATE/UT", "DISTRICT", "YEAR", "MURDER", "RAPE",
                  "THEFT", "ROBBERY", "TOTAL IPC CRIMES"}
    if sort_by not in valid_sort:
        sort_by = "TOTAL IPC CRIMES"

    filtered = df.copy()
    if state_filter:
        filtered = filtered[filtered["STATE/UT"] == state_filter]
    if year_filter:
        try:
            filtered = filtered[filtered["YEAR"] == int(year_filter)]
        except ValueError:
            pass
    if district_search:
        filtered = filtered[filtered["DISTRICT"].str.contains(district_search, na=False)]

    filtered  = filtered.sort_values(sort_by, ascending=(sort_order != "desc"))
    total     = len(filtered)
    start     = (page - 1) * per_page
    disp_cols = ["STATE/UT", "DISTRICT", "YEAR", "MURDER", "RAPE",
                 "THEFT", "ROBBERY", "TOTAL IPC CRIMES"]
    # Only keep columns that actually exist in the dataframe
    disp_cols = [c for c in disp_cols if c in filtered.columns]
    page_data = filtered.iloc[start:start + per_page][disp_cols].to_dict("records")

    return render_template(
        "data.html",
        rows=page_data,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=max(1, (total + per_page - 1) // per_page),
        states=STATES,
        years=YEARS,
        state_filter=state_filter,
        year_filter=year_filter,
        district_search=district_search,
        sort_by=sort_by,
        sort_order=sort_order,
    )


# ─────────────────────────────────────────────────────────────
@app.route("/data/export")
def data_export():
    state_filter    = request.args.get("state",           "").upper().strip()
    year_filter     = request.args.get("year",            "")
    district_search = request.args.get("district_search", "").upper().strip()

    filtered = df.copy()
    if state_filter:
        filtered = filtered[filtered["STATE/UT"] == state_filter]
    if year_filter:
        try:
            filtered = filtered[filtered["YEAR"] == int(year_filter)]
        except ValueError:
            pass
    if district_search:
        filtered = filtered[filtered["DISTRICT"].str.contains(district_search, na=False)]

    filtered = filtered.head(10_000)
    buf = io.StringIO()
    filtered.to_csv(buf, index=False)
    buf.seek(0)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype="text/csv",
        download_name=f"crime_data_{ts}.csv",
        as_attachment=True,
    )


# ─────────────────────────────────────────────────────────────
@app.route("/about")
def about():
    importances = analytics.get("feature_importances", {})
    if not importances and hasattr(model, "feature_importances_"):
        feat_names  = ["Year", "State", "District", "Crime Type"]
        importances = {n: round(float(v) * 100, 1)
                       for n, v in zip(feat_names, model.feature_importances_)}

    avg_crimes_per_year = int(round(
        df.groupby("YEAR")["TOTAL IPC CRIMES"].sum().mean()
    ))

    return render_template(
        "about.html",
        model_r2=round(metrics.get("r2", 0) * 100, 1),
        model_mae=round(metrics.get("mae", 0), 1),
        model_rmse=round(metrics.get("rmse", 0), 1),
        total_records=f"{len(df):,}",
        num_states=len(STATES),
        crime_types=len(CRIME_COLS),
        feature_importances=json.dumps(importances),
        avg_crimes_per_year=f"{avg_crimes_per_year:,}",
        num_districts=int(df["DISTRICT"].nunique()),
    )


# ══════════════════════════════════════════════════════════════
#  NEW JSON API ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.route("/api/state_summary")
def api_state_summary():
    state = request.args.get("state", "").upper().strip()
    if not state or state not in STATES:
        return jsonify({"error": f"Unknown state: '{state}'"}), 400

    crime_breakdown = {}
    if state in _state_crime_totals.index:
        crime_breakdown = {
            k: int(v) for k, v in
            _state_crime_totals.loc[state].sort_values(ascending=False).items()
        }

    year_trend_state = {
        str(yr): int(tot)
        for yr, tot in df[df["STATE/UT"] == state]
        .groupby("YEAR")["TOTAL IPC CRIMES"].sum().items()
    }

    return jsonify({
        "state":           state,
        "crime_breakdown": crime_breakdown,
        "year_trend":      year_trend_state,
        "districts":       DISTRICTS.get(state, []),
    })


@app.route("/api/hotspots")
def api_hotspots():
    crime = request.args.get("crime", "MURDER")
    year  = request.args.get("year",  None)
    n     = min(int(request.args.get("n", 10)), 50)

    if crime not in CRIME_COLS:
        return jsonify({"error": f"Unknown crime type: '{crime}'"}), 400

    filtered = df.copy()
    if year:
        try:
            filtered = filtered[filtered["YEAR"] == int(year)]
        except ValueError:
            return jsonify({"error": "Invalid year"}), 400

    result = (
        filtered.groupby(["STATE/UT", "DISTRICT"])[crime]
        .sum().nlargest(n).reset_index()
        .rename(columns={"STATE/UT": "state", "DISTRICT": "district", crime: "total"})
    )
    return jsonify({"crime": crime, "year": year or "all",
                    "hotspots": result.to_dict("records")})


@app.route("/api/compare")
def api_compare():
    state = request.args.get("state", "").upper().strip()
    d1    = request.args.get("d1",    "").upper().strip()
    d2    = request.args.get("d2",    "").upper().strip()
    crime = request.args.get("crime", "MURDER")

    if crime not in encoders["crime"].classes_:
        return jsonify({"error": f"Unknown crime: '{crime}'"}), 400

    result = {}
    for district in (d1, d2):
        if district not in encoders["district"].classes_:
            result[district] = {"error": f"Unknown district: '{district}'"}
            continue
        hist = {}
        sub  = df[(df["STATE/UT"] == state) & (df["DISTRICT"] == district)]
        if crime in df.columns:
            for _, row in sub.iterrows():
                hist[int(row["YEAR"])] = int(row[crime])

        fc25, _, _ = predict_count(2025, state, district, crime)
        fc30, _, _ = predict_count(2030, state, district, crime)
        result[district] = {
            "historical":    {str(k): v for k, v in hist.items()},
            "forecast_2025": fc25,
            "forecast_2030": fc30,
            "risk_2025":     risk_label(fc25)[0],
        }
    return jsonify({"state": state, "crime": crime, "comparison": result})


@app.route("/api/trend")
def api_trend():
    state    = request.args.get("state",    "").upper().strip()
    district = request.args.get("district", "").upper().strip()
    crime    = request.args.get("crime",    "MURDER")

    for label, val, enc in [
        ("state",    state,    encoders["state"]),
        ("district", district, encoders["district"]),
        ("crime",    crime,    encoders["crime"]),
    ]:
        if val not in enc.classes_:
            return jsonify({"error": f"Unknown {label}: '{val}'"}), 400

    historical = {}
    sub = df[(df["STATE/UT"] == state) & (df["DISTRICT"] == district)]
    if crime in df.columns:
        for _, row in sub.iterrows():
            historical[int(row["YEAR"])] = int(row[crime])

    forecast = forecast_series(state, district, crime)
    combined = {**{str(k): v for k, v in historical.items()},
                **{str(k): v for k, v in forecast.items()}}

    return jsonify({
        "state": state, "district": district, "crime": crime,
        "historical": {str(k): v for k, v in historical.items()},
        "forecast":   {str(k): v for k, v in forecast.items()},
        "combined":   combined,
    })


# ══════════════════════════════════════════════════════════════
#  PDF REPORT
# ══════════════════════════════════════════════════════════════

@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    data       = request.get_json(silent=True) or {}
    state      = data.get("state",      "ALL")
    district   = data.get("district",   "ALL")
    crime_type = data.get("crime_type", "TOTAL IPC CRIMES")
    year       = int(data.get("year",   2013))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=0.75*inch, leftMargin=0.75*inch,
                            topMargin=0.75*inch,  bottomMargin=0.75*inch)
    styles   = getSampleStyleSheet()
    title_s  = ParagraphStyle("T",  parent=styles["Title"],   fontSize=20, spaceAfter=6,
                               textColor=colors.HexColor("#1e3a8a"))
    h2       = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceAfter=4,
                               textColor=colors.HexColor("#1e40af"))
    body     = styles["Normal"]
    note_s   = ParagraphStyle("N",  parent=styles["Normal"],  fontSize=8,
                               textColor=colors.gray)

    def mk_table(rows, widths):
        t = Table(rows, colWidths=widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),  (-1,0),  colors.HexColor("#1e3a8a")),
            ("TEXTCOLOR",      (0,0),  (-1,0),  colors.white),
            ("FONTNAME",       (0,0),  (-1,0),  "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0,1),  (-1,-1),
             [colors.HexColor("#f0f4ff"), colors.white]),
            ("GRID",           (0,0),  (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ("FONTSIZE",       (0,0),  (-1,-1), 10),
            ("PADDING",        (0,0),  (-1,-1), 6),
        ]))
        return t

    story = [
        Paragraph("CrimeWatch Analytics — Report", title_s),
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')} | "
            "NCRB India 2001–2013 | XGBoost Regressor", note_s,
        ),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1e3a8a")),
        Spacer(1, 0.15*inch),
        Paragraph("Query Parameters", h2),
        mk_table([["Field","Value"],["State",state],["District",district],
                  ["Crime Type",crime_type],["Year",str(year)]],
                 [2*inch, 4*inch]),
        Spacer(1, 0.2*inch),
    ]

    if state != "ALL" and district != "ALL":
        count, low, high = predict_count(year, state, district, crime_type)
        label, _         = risk_label(count)
        forecast         = forecast_series(state, district, crime_type)
        summary          = generate_ai_summary(
            state, district, crime_type, year, count, forecast, low, high)

        hist_rows = [["Year", f"Actual {crime_type}"]]
        hist_sub  = df[(df["STATE/UT"] == state) & (df["DISTRICT"] == district)]
        if crime_type in df.columns:
            for _, row in hist_sub.sort_values("YEAR").iterrows():
                hist_rows.append([str(int(row["YEAR"])), str(int(row[crime_type]))])

        story += [
            Paragraph("Prediction Result", h2),
            mk_table([["Metric","Value"],
                      ["Predicted Cases", str(count) if count is not None else "N/A"],
                      ["90% Confidence", f"{low} – {high}" if low is not None else "N/A"],
                      ["Risk Level", label],
                      ["Model R²", f"{round(metrics.get('r2',0)*100,1)}%"],
                      ["MAE", str(round(metrics.get('mae',0),1))],
                      ["RMSE", str(round(metrics.get('rmse',0),1))]],
                     [2.5*inch, 3.5*inch]),
            Spacer(1, 0.2*inch),
        ]

        if len(hist_rows) > 1:
            story += [
                Paragraph("Historical Actuals (2001–2013)", h2),
                mk_table(hist_rows, [2*inch, 4*inch]),
                Spacer(1, 0.2*inch),
            ]

        fc_rows = [["Year","Predicted"]] + [
            [str(yr), str(forecast.get(yr,"N/A"))] for yr in range(2026, 2031)]
        story += [
            Paragraph("5-Year Forecast (2026–2030)", h2),
            mk_table(fc_rows, [2*inch, 4*inch]),
            Spacer(1, 0.2*inch),
            Paragraph("AI Analysis Summary", h2),
        ]
        for line in summary.split("\n"):
            if line.strip():
                story += [Paragraph(line, body), Spacer(1, 0.05*inch)]

    # Hotspots table
    hs = (df.groupby(["STATE/UT","DISTRICT"])["TOTAL IPC CRIMES"]
           .sum().nlargest(10).reset_index())
    hs_rows = [["#","State","District","Total IPC Crimes"]]
    for i,(_, r) in enumerate(hs.iterrows(), 1):
        hs_rows.append([str(i), r["STATE/UT"], r["DISTRICT"],
                        f"{int(r['TOTAL IPC CRIMES']):,}"])

    # Top states table
    ts = df.groupby("STATE/UT")["TOTAL IPC CRIMES"].sum().nlargest(10).reset_index()
    ts_rows = [["#","State","Total IPC Crimes"]]
    for i,(_, r) in enumerate(ts.iterrows(), 1):
        ts_rows.append([str(i), r["STATE/UT"], f"{int(r['TOTAL IPC CRIMES']):,}"])

    story += [
        Spacer(1, 0.2*inch),
        Paragraph("Top 10 District Hotspots", h2),
        mk_table(hs_rows, [0.4*inch, 2.3*inch, 2*inch, 1.5*inch]),
        Spacer(1, 0.2*inch),
        Paragraph("Top 10 States by Total IPC Crimes", h2),
        mk_table(ts_rows, [0.5*inch, 3.5*inch, 2.5*inch]),
        Spacer(1, 0.3*inch),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")),
        Paragraph(
            f"© CrimeWatch Analytics | NCRB India | "
            f"Generated {datetime.now().strftime('%d %b %Y')}", note_s),
    ]

    doc.build(story)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf",
                     download_name=f"crime_report_{state}_{year}.pdf",
                     as_attachment=True)


# ══════════════════════════════════════════════════════════════
#  HEALTH + ERROR HANDLERS
# ══════════════════════════════════════════════════════════════

@app.route("/health")
def health():
    return jsonify({
        "status":  "ok",
        "model":   "loaded",
        "records": len(df),
        "time":    datetime.utcnow().isoformat() + "Z",
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "code": 404}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed", "code": 405}), 405


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Unhandled 500")
    return jsonify({"error": "Internal server error", "code": 500}), 500


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info(f"Starting CrimeWatch on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)
