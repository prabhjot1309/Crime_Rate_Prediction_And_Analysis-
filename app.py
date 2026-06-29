"""
CrimeWatch Analytics — Enhanced Flask Application
================================================
Improvements per route:
  /                  → richer dashboard stats (district count, year range, avg per district)
  /get_districts     → returns district list AND crime summary counts for that state
  /predict           → smarter validation, year-range guard, confidence interval, % change
  /predictions       → new query params: pre-fill state/district/crime from URL
  /reports           → per-crime breakdown, YoY trend, district-level hotspot table
  /data              → search by district name, multi-col sorting, exportable CSV slice
  /about             → feature importances, dataset stats, model comparison
  /download_pdf      → richer PDF: historical table, forecast chart as ASCII, confidence
  /api/state_summary → NEW: JSON endpoint returning full state crime breakdown
  /api/hotspots      → NEW: top N district hotspots for a given crime/year
  /api/compare       → NEW: side-by-side district comparison for a crime type
  /api/trend         → NEW: multi-year trend for a district+crime (historical + forecast)
  /health            → NEW: health-check endpoint (useful for Docker/k8s probes)
"""

from flask import Flask, render_template, request, jsonify, send_file, abort
import pandas as pd
import numpy as np
import joblib
import json
import os
import io
import logging
from datetime import datetime
from functools import lru_cache

# ── PDF ──────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)

# ── LOGGING SETUP ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s : %(message)s",
)
logger = logging.getLogger("crimewatch")

np.random.seed(42)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ── PATHS ─────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "model_files")


# ── LOAD ASSETS ───────────────────────────────────────────────
def _load_assets():
    """Load all heavy assets once at startup and return them."""
    logger.info("Loading crime.csv …")
    raw = pd.read_csv(os.path.join(BASE, "crime.csv"))
    raw.fillna(0, inplace=True)
    raw["STATE/UT"] = raw["STATE/UT"].str.upper().str.strip()
    raw["DISTRICT"] = raw["DISTRICT"].str.upper().str.strip()
    df = raw[raw["DISTRICT"] != "TOTAL"].copy()

    crime_cols = [c for c in df.columns
                  if c not in ["STATE/UT", "DISTRICT", "YEAR", "TOTAL IPC CRIMES"]]

    logger.info("Loading model artefacts …")
    model     = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    encoders  = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    metrics   = joblib.load(os.path.join(MODEL_DIR, "metrics.pkl"))
    with open(os.path.join(MODEL_DIR, "analytics.json")) as fh:
        analytics = json.load(fh)

    logger.info(f"Assets loaded — {len(df):,} rows, {len(crime_cols)} crime types.")
    return df, crime_cols, model, encoders, metrics, analytics


df, CRIME_COLS, model, encoders, metrics, analytics = _load_assets()

STATES      = analytics["states"]
DISTRICTS   = analytics["districts_by_state"]
YEARS       = analytics["years"]           # 2001-2013 (actual dataset)
FULL_YEARS  = list(range(2001, 2031))      # 2001-2030 (historical + forecast)

# Pre-compute a quick per-state crime series for fast lookups
_state_crime_totals = df.groupby("STATE/UT")[CRIME_COLS].sum()
_district_crime_totals = df.groupby(["STATE/UT", "DISTRICT"])[CRIME_COLS].sum()


# ── HELPERS ───────────────────────────────────────────────────

def safe_encode(encoder, value):
    """Return encoded integer or None if value is unseen."""
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return None


def predict_count(year: int, state: str, district: str, crime_type: str):
    """Return (predicted_count, low_bound, high_bound) or (None, None, None)."""
    s = safe_encode(encoders["state"], state)
    d = safe_encode(encoders["district"], district)
    c = safe_encode(encoders["crime"], crime_type)
    if any(v is None for v in [s, d, c]):
        return None, None, None

    val = float(model.predict([[year, s, d, c]])[0])
    val = max(0.0, val)

    # Approximate 90 % confidence interval using MAE as a proxy for ±1.65σ
    half_ci = metrics["mae"] * 1.65
    low  = max(0.0, round(val - half_ci, 1))
    high = round(val + half_ci, 1)
    return round(val, 1), low, high


def forecast_series(state: str, district: str, crime_type: str,
                    from_year: int = 2014, to_year: int = 2030) -> dict:
    """Return {year: predicted_count} for the requested range."""
    s = safe_encode(encoders["state"], state)
    d = safe_encode(encoders["district"], district)
    c = safe_encode(encoders["crime"], crime_type)
    if any(v is None for v in [s, d, c]):
        return {}

    results = {}
    for yr in range(from_year, to_year + 1):
        val = float(model.predict([[yr, s, d, c]])[0])
        results[yr] = max(0.0, round(val, 1))
    return results


def risk_label(count):
    """Classify a crime count into a risk band."""
    if count is None:
        return "Unknown", "gray"
    if count == 0:
        return "None", "green"
    if count <= 35:
        return "Low", "green"
    if count <= 65:
        return "Medium", "yellow"
    if count <= 150:
        return "High", "orange"
    return "Very High", "red"


def pct_change(old, new):
    """Safe percentage change."""
    if old is None or old == 0:
        return None
    return round((new - old) / old * 100, 1)


def generate_ai_summary(state, district, crime_type, year, count, forecast,
                         low=None, high=None):
    """Generate a multi-line textual AI analysis summary."""
    label, _ = risk_label(count)
    trend_vals = list(forecast.values())

    direction = "stable"
    if len(trend_vals) >= 3:
        direction = "rising" if trend_vals[-1] > trend_vals[0] else "declining"

    # YoY change between the two most recent forecast years
    yoy = None
    if len(trend_vals) >= 2:
        yoy = pct_change(trend_vals[-2], trend_vals[-1])

    lines = [
        f"📍 District: {district}, {state}",
        f"📅 Year: {year} | Crime: {crime_type}",
        "",
        f"🔍 Risk Level: {label}",
        f"📊 Predicted Cases: {count if count is not None else 'N/A'}",
    ]

    if low is not None and high is not None:
        lines.append(f"   Confidence Range (90%): {low} – {high} cases")

    lines += [
        "",
        f"📈 Forecast Trend (2014–2030): {direction.upper()}",
    ]

    if yoy is not None:
        arrow = "↑" if yoy > 0 else "↓"
        lines.append(f"   Year-on-Year Change (latest): {arrow} {abs(yoy)}%")

    if direction == "rising":
        lines.append("⚠️  Increasing trend detected — enhanced patrolling may help.")
    elif direction == "declining":
        lines.append("✅ Declining trend — current measures appear effective.")
    else:
        lines.append("➡️  Trend is stable — consistent enforcement recommended.")

    # Flag if state is a top-3 contributor
    if crime_type in df.columns:
        top3 = _state_crime_totals[crime_type].nlargest(3).index.tolist()
        if state in top3:
            lines.append(f"🚨 {state} is among the top 3 states for {crime_type}.")

    # Flag if district itself is a district-level hotspot
    if crime_type in df.columns:
        key = (state, district)
        if key in _district_crime_totals.index:
            dist_total = _district_crime_totals.loc[key, crime_type]
            state_total = _state_crime_totals.loc[state, crime_type] if state in _state_crime_totals.index else 0
            if state_total > 0:
                share = round(dist_total / state_total * 100, 1)
                lines.append(f"📌 {district} accounts for ~{share}% of {state}'s {crime_type} cases (2001-2013).")

    return "\n".join(lines)


def _validate_predict_inputs(data: dict):
    """Return (year, state, district, crime_type) or raise ValueError."""
    year = int(data.get("year", 2013))
    state = str(data.get("state", "")).upper().strip()
    district = str(data.get("district", "")).upper().strip()
    crime_type = str(data.get("crime_type", ""))

    if not state or not district or not crime_type:
        raise ValueError(f"Missing field — state:{state!r} district:{district!r} crime:{crime_type!r}")
    if year < 2001 or year > 2030:
        raise ValueError(f"Year {year} is outside the supported range 2001–2030.")
    if state not in encoders["state"].classes_:
        raise ValueError(f"Unknown state: {state!r}. Please check spelling.")
    if district not in encoders["district"].classes_:
        raise ValueError(f"Unknown district: {district!r}. Please check spelling.")
    if crime_type not in encoders["crime"].classes_:
        raise ValueError(f"Unknown crime type: {crime_type!r}.")

    return year, state, district, crime_type


# ── ROUTES ────────────────────────────────────────────────────

@app.route("/")
def index():
    """
    Dashboard — enhanced with extra stats:
    • avg crimes per district
    • district count across dataset
    • dataset year range
    • crime type count
    """
    total_crimes   = int(df["TOTAL IPC CRIMES"].sum())
    district_count = df["DISTRICT"].nunique()
    avg_per_dist   = round(total_crimes / district_count) if district_count else 0
    year_min, year_max = int(df["YEAR"].min()), int(df["YEAR"].max())

    # Top crime type (useful for hero stat on dashboard)
    crime_totals = {c: int(df[c].sum()) for c in CRIME_COLS}
    top_crime = max(crime_totals, key=crime_totals.get)

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
        model_r2=round(metrics["r2"] * 100, 1),
        model_mae=round(metrics["mae"], 1),
    )


@app.route("/get_districts")
def get_districts():
    """
    Enhanced: returns districts AND optional crime-level summary stats
    for the requested state so the frontend can show mini-stats.

    Query params:
      state    (required)
      include_stats  (optional, bool) – if "1", include top-3 crime totals
    """
    state = request.args.get("state", "").upper().strip()
    include_stats = request.args.get("include_stats", "0") == "1"

    districts = DISTRICTS.get(state, [])

    if not include_stats:
        return jsonify(districts)

    # Build a quick per-district total for this state
    state_df = df[df["STATE/UT"] == state]
    dist_totals = (
        state_df.groupby("DISTRICT")["TOTAL IPC CRIMES"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    )

    top_crimes_for_state = {}
    if state in _state_crime_totals.index:
        top_crimes_for_state = (
            _state_crime_totals.loc[state]
            .sort_values(ascending=False)
            .head(5)
            .to_dict()
        )

    return jsonify({
        "districts": districts,
        "top_districts_by_crime": dist_totals,
        "top_crimes": top_crimes_for_state,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Enhanced prediction endpoint:
    • Input validation extracted to helper (cleaner try/except)
    • Returns confidence interval (low, high)
    • Returns year-over-year % change in forecast
    • Returns 'is_forecast' flag (True if year > 2013)
    • Returns district's historical avg for context
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        year, state, district, crime_type = _validate_predict_inputs(data)

    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    try:
        count, low, high = predict_count(year, state, district, crime_type)
        label, color = risk_label(count)

        from_year = max(year, 2014)
        forecast  = forecast_series(state, district, crime_type,
                                    from_year=from_year, to_year=2030)

        # Historical actuals from dataset
        hist_df = df[(df["STATE/UT"] == state) & (df["DISTRICT"] == district)]
        historical = {}
        if crime_type in df.columns:
            for _, row in hist_df.iterrows():
                historical[int(row["YEAR"])] = int(row[crime_type])

        # District historical average (for context)
        hist_avg = round(np.mean(list(historical.values())), 1) if historical else None

        # YoY % change between last two forecast years
        fvals = list(forecast.values())
        yoy_pct = pct_change(fvals[-2], fvals[-1]) if len(fvals) >= 2 else None

        summary = generate_ai_summary(
            state, district, crime_type, year, count, forecast, low, high
        )

        return jsonify({
            "count":       count,
            "low":         low,
            "high":        high,
            "label":       label,
            "color":       color,
            "is_forecast": year > 2013,
            "hist_avg":    hist_avg,
            "yoy_pct":     yoy_pct,
            "summary":     summary,
            "forecast":    {str(k): v for k, v in forecast.items()},
            "historical":  {str(k): v for k, v in historical.items()},
        })

    except Exception as e:
        logger.exception("Error in /predict")
        return jsonify({"error": str(e)}), 500


@app.route("/predictions")
def predictions():
    """
    Predictions page — now accepts optional URL query params to pre-fill
    the form (state, district, crime, year) so users can share deep links.
    """
    prefill = {
        "state":      request.args.get("state", "").upper().strip(),
        "district":   request.args.get("district", "").upper().strip(),
        "crime":      request.args.get("crime", ""),
        "year":       request.args.get("year", ""),
    }
    return render_template(
        "predictions.html",
        states=STATES,
        crime_cols=CRIME_COLS,
        years=FULL_YEARS,
        prefill=prefill,
    )


@app.route("/reports")
def reports():
    """
    Reports page — enhanced with:
    • per-crime totals breakdown (all CRIME_COLS)
    • YoY trend (year-by-year delta)
    • district-level hotspot table (top 10 across whole dataset)
    • state-level rank table extended to top 20
    """
    # Top 20 states
    state_summary = (
        df.groupby("STATE/UT")
        .agg(
            total_crimes=("TOTAL IPC CRIMES", "sum"),
            years_count=("YEAR", "nunique"),
            districts=("DISTRICT", "nunique"),
        )
        .reset_index()
        .sort_values("total_crimes", ascending=False)
        .head(20)
    )
    state_summary["rank"] = range(1, len(state_summary) + 1)

    # Year-over-Year trend delta
    yt = analytics["year_trend"]
    years_list  = yt["years"]
    totals_list = yt["totals"]
    yoy_deltas  = [0] + [
        round(totals_list[i] - totals_list[i - 1])
        for i in range(1, len(totals_list))
    ]

    # Top 10 district hotspots (all-time total IPC crimes)
    hotspots = (
        df.groupby(["STATE/UT", "DISTRICT"])["TOTAL IPC CRIMES"]
        .sum()
        .nlargest(10)
        .reset_index()
    )
    hotspot_rows = hotspots.rename(
        columns={"STATE/UT": "state", "DISTRICT": "district",
                 "TOTAL IPC CRIMES": "total"}
    ).to_dict("records")

    # Per-crime totals for a bar chart
    crime_totals = sorted(
        [{"crime": c, "total": int(df[c].sum())} for c in CRIME_COLS],
        key=lambda x: x["total"],
        reverse=True,
    )

    return render_template(
        "reports.html",
        state_rows=state_summary.to_dict("records"),
        hotspot_rows=hotspot_rows,
        crime_totals=json.dumps(crime_totals),
        yoy_years=json.dumps(years_list),
        yoy_deltas=json.dumps(yoy_deltas),
        model_r2=round(metrics["r2"] * 100, 1),
        model_mae=round(metrics["mae"], 1),
        model_rmse=round(metrics["rmse"], 1),
        top_crimes=json.dumps(analytics["top_crimes"]),
        state_data=json.dumps(analytics["state_data"]),
        year_trend=json.dumps(analytics["year_trend"]),
    )


@app.route("/data")
def data_page():
    """
    Data explorer — enhanced with:
    • district name search (partial match)
    • multi-column sort (sort_by + order params)
    • CSV export link (handled by /data/export)
    • dynamic per_page (10/25/50/100)
    """
    page        = max(1, int(request.args.get("page", 1)))
    per_page    = int(request.args.get("per_page", 50))
    if per_page not in (10, 25, 50, 100):
        per_page = 50

    state_filter    = request.args.get("state", "").upper().strip()
    year_filter     = request.args.get("year", "")
    district_search = request.args.get("district_search", "").upper().strip()
    sort_by         = request.args.get("sort_by", "TOTAL IPC CRIMES")
    sort_order      = request.args.get("order", "desc")

    valid_sort_cols = {"STATE/UT", "DISTRICT", "YEAR", "MURDER", "RAPE",
                       "THEFT", "ROBBERY", "TOTAL IPC CRIMES"}
    if sort_by not in valid_sort_cols:
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

    ascending = sort_order != "desc"
    filtered = filtered.sort_values(sort_by, ascending=ascending)

    total   = len(filtered)
    start   = (page - 1) * per_page
    display_cols = ["STATE/UT", "DISTRICT", "YEAR", "MURDER", "RAPE",
                    "THEFT", "ROBBERY", "TOTAL IPC CRIMES"]
    page_data = filtered.iloc[start:start + per_page][display_cols].to_dict("records")

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


@app.route("/data/export")
def data_export():
    """
    NEW — Export filtered data as a CSV download.
    Accepts the same filter params as /data.
    """
    state_filter    = request.args.get("state", "").upper().strip()
    year_filter     = request.args.get("year", "")
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

    # Cap export at 10,000 rows to avoid memory issues
    filtered = filtered.head(10_000)

    buf = io.StringIO()
    filtered.to_csv(buf, index=False)
    buf.seek(0)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype="text/csv",
        download_name=f"crime_data_export_{ts}.csv",
        as_attachment=True,
    )


@app.route("/about")
def about():
    """
    About page — enhanced with:
    • feature importances from the XGBoost model
    • dataset stats breakdown (crimes per year avg)
    • model comparison note (MAE, RMSE, R²)
    """
    importances = {}
    if hasattr(model, "feature_importances_"):
        feat_names = ["Year", "State", "District", "Crime Type"]
        importances = dict(zip(feat_names, [
            round(float(v) * 100, 1) for v in model.feature_importances_
        ]))

    avg_crimes_per_year = round(df.groupby("YEAR")["TOTAL IPC CRIMES"].sum().mean())

    return render_template(
        "about.html",
        model_r2=round(metrics["r2"] * 100, 1),
        model_mae=round(metrics["mae"], 1),
        model_rmse=round(metrics["rmse"], 1),
        total_records=f"{len(df):,}",
        num_states=len(STATES),
        crime_types=len(CRIME_COLS),
        feature_importances=json.dumps(importances),
        avg_crimes_per_year=f"{avg_crimes_per_year:,}",
        num_districts=df["DISTRICT"].nunique(),
    )


# ── NEW API ENDPOINTS ─────────────────────────────────────────

@app.route("/api/state_summary")
def api_state_summary():
    """
    NEW JSON API — full crime breakdown for a given state.
    Query: ?state=UTTAR PRADESH
    Returns: {crime_type: total, …} sorted descending, plus district list.
    """
    state = request.args.get("state", "").upper().strip()
    if not state or state not in STATES:
        return jsonify({"error": f"Unknown state: {state!r}"}), 400

    if state in _state_crime_totals.index:
        totals = _state_crime_totals.loc[state].sort_values(ascending=False)
        crime_breakdown = {k: int(v) for k, v in totals.items()}
    else:
        crime_breakdown = {}

    year_trend_state = (
        df[df["STATE/UT"] == state]
        .groupby("YEAR")["TOTAL IPC CRIMES"]
        .sum()
        .to_dict()
    )

    return jsonify({
        "state": state,
        "crime_breakdown": crime_breakdown,
        "year_trend": {str(k): int(v) for k, v in year_trend_state.items()},
        "districts": DISTRICTS.get(state, []),
    })


@app.route("/api/hotspots")
def api_hotspots():
    """
    NEW JSON API — top N district hotspots for a crime type and year.
    Query: ?crime=MURDER&year=2010&n=10
    """
    crime = request.args.get("crime", "MURDER")
    n     = min(int(request.args.get("n", 10)), 50)
    year  = request.args.get("year", None)

    if crime not in CRIME_COLS:
        return jsonify({"error": f"Unknown crime type: {crime!r}"}), 400

    filtered = df.copy()
    if year:
        try:
            filtered = filtered[filtered["YEAR"] == int(year)]
        except ValueError:
            return jsonify({"error": "Invalid year"}), 400

    hotspots = (
        filtered.groupby(["STATE/UT", "DISTRICT"])[crime]
        .sum()
        .nlargest(n)
        .reset_index()
        .rename(columns={"STATE/UT": "state", "DISTRICT": "district", crime: "total"})
    )

    return jsonify({
        "crime":    crime,
        "year":     year or "all",
        "hotspots": hotspots.to_dict("records"),
    })


@app.route("/api/compare")
def api_compare():
    """
    NEW JSON API — side-by-side district comparison for a crime type.
    Query: ?state=UTTAR PRADESH&d1=AGRA&d2=LUCKNOW&crime=THEFT
    Returns historical totals + 2025 forecast for both districts.
    """
    state  = request.args.get("state", "").upper().strip()
    d1     = request.args.get("d1", "").upper().strip()
    d2     = request.args.get("d2", "").upper().strip()
    crime  = request.args.get("crime", "MURDER")

    if crime not in encoders["crime"].classes_:
        return jsonify({"error": f"Unknown crime: {crime!r}"}), 400

    result = {}
    for district in (d1, d2):
        if district not in encoders["district"].classes_:
            result[district] = {"error": f"Unknown district: {district!r}"}
            continue

        hist = {}
        sub = df[(df["STATE/UT"] == state) & (df["DISTRICT"] == district)]
        if crime in df.columns:
            for _, row in sub.iterrows():
                hist[int(row["YEAR"])] = int(row[crime])

        fc_2025, _, _ = predict_count(2025, state, district, crime)
        fc_2030, _, _ = predict_count(2030, state, district, crime)

        result[district] = {
            "historical":    {str(k): v for k, v in hist.items()},
            "forecast_2025": fc_2025,
            "forecast_2030": fc_2030,
            "risk_2025":     risk_label(fc_2025)[0],
        }

    return jsonify({"state": state, "crime": crime, "comparison": result})


@app.route("/api/trend")
def api_trend():
    """
    NEW JSON API — full multi-year trend (historical + forecast) for one district+crime.
    Query: ?state=DELHI&district=CENTRAL&crime=ROBBERY
    """
    state    = request.args.get("state", "").upper().strip()
    district = request.args.get("district", "").upper().strip()
    crime    = request.args.get("crime", "MURDER")

    for label, val, encoder in [
        ("state",    state,    encoders["state"]),
        ("district", district, encoders["district"]),
        ("crime",    crime,    encoders["crime"]),
    ]:
        if val not in encoder.classes_:
            return jsonify({"error": f"Unknown {label}: {val!r}"}), 400

    historical = {}
    sub = df[(df["STATE/UT"] == state) & (df["DISTRICT"] == district)]
    if crime in df.columns:
        for _, row in sub.iterrows():
            historical[int(row["YEAR"])] = int(row[crime])

    forecast = forecast_series(state, district, crime)

    combined = {**{str(k): v for k, v in historical.items()},
                **{str(k): v for k, v in forecast.items()}}

    return jsonify({
        "state":      state,
        "district":   district,
        "crime":      crime,
        "historical": {str(k): v for k, v in historical.items()},
        "forecast":   {str(k): v for k, v in forecast.items()},
        "combined":   combined,
    })


# ── PDF REPORT ────────────────────────────────────────────────

@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    """
    Enhanced PDF report:
    • Historical actuals table (if district selected)
    • 5-year forecast table
    • Confidence interval row
    • Top-10 district hotspots table
    • Top-10 states table
    • Proper footer with generation timestamp
    """
    data       = request.json or {}
    state      = data.get("state", "ALL")
    district   = data.get("district", "ALL")
    crime_type = data.get("crime_type", "TOTAL IPC CRIMES")
    year       = int(data.get("year", 2013))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=0.75 * inch, leftMargin=0.75 * inch,
        topMargin=0.75 * inch,  bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()

    title_sty = ParagraphStyle(
        "T", parent=styles["Title"], fontSize=20, spaceAfter=6,
        textColor=colors.HexColor("#1e3a8a"),
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=13, spaceAfter=4,
        textColor=colors.HexColor("#1e40af"),
    )
    body = styles["Normal"]
    note_sty = ParagraphStyle(
        "Note", parent=styles["Normal"], fontSize=8, textColor=colors.gray,
    )

    def make_table(rows, col_widths):
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#1e3a8a")),
            ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#f0f4ff"), colors.white]),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("FONTSIZE",     (0, 0), (-1, -1), 10),
            ("PADDING",      (0, 0), (-1, -1), 6),
        ]))
        return t

    story = [
        Paragraph("CrimeWatch Analytics — Prediction Report", title_sty),
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}  |  "
            f"Dataset: NCRB India 2001–2013  |  Model: XGBoost Regressor",
            note_sty,
        ),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1e3a8a")),
        Spacer(1, 0.15 * inch),
        Paragraph("Query Parameters", h2),
        make_table(
            [["Field", "Value"],
             ["State", state], ["District", district],
             ["Crime Type", crime_type], ["Year", str(year)]],
            [2 * inch, 4 * inch],
        ),
        Spacer(1, 0.2 * inch),
    ]

    if state != "ALL" and district != "ALL":
        count, low, high = predict_count(year, state, district, crime_type)
        label, _         = risk_label(count)
        forecast         = forecast_series(state, district, crime_type)
        summary          = generate_ai_summary(
            state, district, crime_type, year, count, forecast, low, high
        )

        # Historical table
        hist_df = df[(df["STATE/UT"] == state) & (df["DISTRICT"] == district)]
        hist_rows = [["Year", f"Actual {crime_type}"]]
        if crime_type in df.columns:
            for _, row in hist_df.sort_values("YEAR").iterrows():
                hist_rows.append([str(int(row["YEAR"])), str(int(row[crime_type]))])

        # Forecast table
        fc_rows = [["Year", "Predicted Cases"]] + [
            [str(yr), str(forecast.get(yr, "N/A"))] for yr in range(2026, 2031)
        ]

        story += [
            Paragraph("Prediction Result", h2),
            make_table(
                [["Metric", "Value"],
                 ["Predicted Cases", str(count) if count is not None else "N/A"],
                 ["90% Confidence Range",
                  f"{low} – {high}" if (low is not None and high is not None) else "N/A"],
                 ["Risk Level", label],
                 ["Model R² Score", f"{round(metrics['r2'] * 100, 1)}%"],
                 ["Mean Absolute Error", str(round(metrics["mae"], 1))],
                 ["RMSE", str(round(metrics["rmse"], 1))]],
                [2.5 * inch, 3.5 * inch],
            ),
            Spacer(1, 0.2 * inch),
        ]

        if len(hist_rows) > 1:
            story += [
                Paragraph("Historical Data (2001–2013)", h2),
                make_table(hist_rows, [2 * inch, 4 * inch]),
                Spacer(1, 0.2 * inch),
            ]

        story += [
            Paragraph("5-Year Forecast (2026–2030)", h2),
            make_table(fc_rows, [2 * inch, 4 * inch]),
            Spacer(1, 0.2 * inch),
            Paragraph("AI Analysis Summary", h2),
        ]

        for line in summary.split("\n"):
            if line.strip():
                story += [Paragraph(line, body), Spacer(1, 0.05 * inch)]

    # Top 10 district hotspots
    hotspot_df = (
        df.groupby(["STATE/UT", "DISTRICT"])["TOTAL IPC CRIMES"]
        .sum()
        .nlargest(10)
        .reset_index()
    )
    hs_rows = [["#", "State", "District", "Total IPC Crimes"]]
    for i, (_, r) in enumerate(hotspot_df.iterrows(), 1):
        hs_rows.append([str(i), r["STATE/UT"], r["DISTRICT"],
                         f"{int(r['TOTAL IPC CRIMES']):,}"])

    # Top 10 states
    top_states = (
        df.groupby("STATE/UT")["TOTAL IPC CRIMES"]
        .sum()
        .nlargest(10)
        .reset_index()
    )
    ts_rows = [["#", "State", "Total IPC Crimes"]]
    for i, (_, r) in enumerate(top_states.iterrows(), 1):
        ts_rows.append([str(i), r["STATE/UT"], f"{int(r['TOTAL IPC CRIMES']):,}"])

    story += [
        Spacer(1, 0.2 * inch),
        Paragraph("Top 10 District Hotspots (All-Time)", h2),
        make_table(hs_rows, [0.4*inch, 2.5*inch, 2*inch, 1.5*inch]),
        Spacer(1, 0.2 * inch),
        Paragraph("Top 10 States by Total IPC Crimes (2001–2013)", h2),
        make_table(ts_rows, [0.5*inch, 3.5*inch, 2.5*inch]),
        Spacer(1, 0.3 * inch),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0")),
        Paragraph(
            "© CrimeWatch Analytics  |  NCRB India 2001–2013  |  XGBoost Regressor  |  "
            f"Generated {datetime.now().strftime('%d %b %Y')}",
            note_sty,
        ),
    ]

    doc.build(story)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/pdf",
        download_name=f"crime_report_{state}_{year}.pdf",
        as_attachment=True,
    )


# ── HEALTH CHECK ──────────────────────────────────────────────

@app.route("/health")
def health():
    """
    NEW — Lightweight health-check endpoint.
    Returns 200 + JSON when app + model are loaded correctly.
    Useful for Docker HEALTHCHECK, Kubernetes liveness probes, etc.
    """
    return jsonify({
        "status":  "ok",
        "model":   "loaded",
        "records": len(df),
        "time":    datetime.utcnow().isoformat() + "Z",
    })


# ── ERROR HANDLERS ────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "code": 404}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed", "code": 405}), 405


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Unhandled 500 error")
    return jsonify({"error": "Internal server error", "code": 500}), 500


# ── ENTRY POINT ───────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info(f"Starting CrimeWatch on port {port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)
