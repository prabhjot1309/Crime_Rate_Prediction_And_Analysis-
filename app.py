from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import json
import os
import io
from datetime import datetime

# ── PDF ──────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)

np.random.seed(42)
app = Flask(__name__)

# ── PATHS ─────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, "model_files")

# ── LOAD ASSETS ───────────────────────────────────────────────
df_raw = pd.read_csv(os.path.join(BASE, "crime.csv"))
df_raw.fillna(0, inplace=True)
df_raw['STATE/UT'] = df_raw['STATE/UT'].str.upper().str.strip()
df_raw['DISTRICT'] = df_raw['DISTRICT'].str.upper().str.strip()
df = df_raw[df_raw['DISTRICT'] != 'TOTAL'].copy()

CRIME_COLS = [c for c in df.columns
              if c not in ['STATE/UT', 'DISTRICT', 'YEAR', 'TOTAL IPC CRIMES']]

model    = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
metrics  = joblib.load(os.path.join(MODEL_DIR, "metrics.pkl"))

with open(os.path.join(MODEL_DIR, "analytics.json")) as f:
    analytics = json.load(f)

STATES     = analytics['states']
DISTRICTS  = analytics['districts_by_state']
YEARS      = analytics['years']               # 2001-2013 (dataset only)
FULL_YEARS = list(range(2001, 2031))          # 2001-2030 (historical + forecast)

# ── HELPERS ───────────────────────────────────────────────────
def safe_encode(encoder, value):
    classes = list(encoder.classes_)
    if value in classes:
        return encoder.transform([value])[0]
    return None

def predict_count(year, state, district, crime_type):
    s = safe_encode(encoders['state'], state)
    d = safe_encode(encoders['district'], district)
    c = safe_encode(encoders['crime'], crime_type)
    if any(v is None for v in [s, d, c]):
        return None
    val = model.predict([[year, s, d, c]])[0]
    return max(0, round(float(val), 1))

def forecast_series(state, district, crime_type, from_year=2014, to_year=2030):
    s = safe_encode(encoders['state'], state)
    d = safe_encode(encoders['district'], district)
    c = safe_encode(encoders['crime'], crime_type)
    if any(v is None for v in [s, d, c]):
        return {}
    results = {}
    for yr in range(from_year, to_year + 1):
        val = model.predict([[yr, s, d, c]])[0]
        results[yr] = max(0, round(float(val), 1))
    return results

def risk_label(count):
    if count is None: return "Unknown", "gray"
    if count == 0:    return "None",    "green"
    if count <= 35:   return "Low",     "green"
    if count <= 65:   return "Medium",  "yellow"
    if count <= 150:  return "High",    "orange"
    return "Very High", "red"

def generate_ai_summary(state, district, crime_type, year, count, forecast):
    label, _ = risk_label(count)
    trend_vals = list(forecast.values())
    if len(trend_vals) >= 3:
        direction = "rising" if trend_vals[-1] > trend_vals[0] else "declining"
    else:
        direction = "stable"

    lines = [
        f"📍 District: {district}, {state}",
        f"📅 Year: {year} | Crime: {crime_type}",
        f"",
        f"🔍 Risk Level: {label}",
        f"📊 Predicted Cases: {count if count is not None else 'N/A'}",
        f"",
        f"📈 Forecast Trend (2014–2030): {direction.upper()}",
    ]
    if direction == "rising":
        lines.append("⚠️  Increasing trend detected. Authorities may consider enhanced patrolling.")
    else:
        lines.append("✅  Declining trend noted. Current measures appear effective.")

    high_states = df.groupby('STATE/UT')[crime_type].sum().nlargest(3).index.tolist() \
                  if crime_type in df.columns else []
    if state in high_states:
        lines.append(f"🚨  {state} is among the top 3 states for {crime_type}.")

    return "\n".join(lines)

# ── ROUTES ────────────────────────────────────────────────────
@app.route('/')
def index():
    total_crimes = int(df['TOTAL IPC CRIMES'].sum())
    return render_template('index.html',
        states=STATES,
        years=FULL_YEARS,           # ← full 2001-2030
        crime_cols=CRIME_COLS,
        year_trend=json.dumps(analytics['year_trend']),
        top_crimes=json.dumps(analytics['top_crimes']),
        state_data=json.dumps(analytics['state_data']),
        total_crimes=f"{total_crimes:,}",
        model_r2=round(metrics['r2'] * 100, 1),
        model_mae=round(metrics['mae'], 1),
    )

@app.route('/get_districts')
def get_districts():
    state = request.args.get('state', '').upper().strip()
    return jsonify(DISTRICTS.get(state, []))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No JSON body received'}), 400

        year       = int(data.get('year', 2013))
        state      = str(data.get('state', '')).upper().strip()
        district   = str(data.get('district', '')).upper().strip()
        crime_type = str(data.get('crime_type', ''))

        if not state or not district or not crime_type:
            return jsonify({'error': f'Missing field — state:{state} district:{district} crime:{crime_type}'}), 400

        # Validate state/district exist in encoders
        if state not in encoders['state'].classes_:
            return jsonify({'error': f'Unknown state: {state}. Check spelling.'}), 400
        if district not in encoders['district'].classes_:
            return jsonify({'error': f'Unknown district: {district}. Check spelling.'}), 400
        if crime_type not in encoders['crime'].classes_:
            return jsonify({'error': f'Unknown crime type: {crime_type}'}), 400

        count        = predict_count(year, state, district, crime_type)
        label, color = risk_label(count)
        forecast     = forecast_series(state, district, crime_type,
                                       from_year=max(year, 2014), to_year=2030)
        summary      = generate_ai_summary(state, district, crime_type, year, count, forecast)

        # Historical actuals from dataset
        hist = df[(df['STATE/UT'] == state) & (df['DISTRICT'] == district)]
        historical = {}
        if crime_type in df.columns:
            for _, row in hist.iterrows():
                historical[int(row['YEAR'])] = int(row[crime_type])

        return jsonify({
            'count':      count,
            'label':      label,
            'color':      color,
            'summary':    summary,
            'forecast':   {str(k): v for k, v in forecast.items()},
            'historical': {str(k): v for k, v in historical.items()},
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predictions')
def predictions():
    return render_template('predictions.html',
        states=STATES,
        crime_cols=CRIME_COLS,
        years=FULL_YEARS,           # ← full 2001-2030
    )

@app.route('/reports')
def reports():
    state_summary = df.groupby('STATE/UT').agg(
        total_crimes=('TOTAL IPC CRIMES', 'sum'),
        years_count=('YEAR', 'nunique'),
        districts=('DISTRICT', 'nunique')
    ).reset_index().sort_values('total_crimes', ascending=False).head(15)

    return render_template('reports.html',
        state_rows=state_summary.to_dict('records'),
        model_r2=round(metrics['r2'] * 100, 1),
        model_mae=round(metrics['mae'], 1),
        model_rmse=round(metrics['rmse'], 1),
        top_crimes=json.dumps(analytics['top_crimes']),
        state_data=json.dumps(analytics['state_data']),
        year_trend=json.dumps(analytics['year_trend']),
    )

@app.route('/data')
def data_page():
    page         = int(request.args.get('page', 1))
    per_page     = 50
    state_filter = request.args.get('state', '')
    year_filter  = request.args.get('year', '')

    filtered = df.copy()
    if state_filter:
        filtered = filtered[filtered['STATE/UT'] == state_filter.upper()]
    if year_filter:
        filtered = filtered[filtered['YEAR'] == int(year_filter)]

    total     = len(filtered)
    start     = (page - 1) * per_page
    page_data = filtered.iloc[start:start + per_page][
        ['STATE/UT', 'DISTRICT', 'YEAR', 'MURDER', 'RAPE',
         'THEFT', 'ROBBERY', 'TOTAL IPC CRIMES']
    ].to_dict('records')

    return render_template('data.html',
        rows=page_data,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=(total // per_page) + 1,
        states=STATES,
        years=YEARS,                # data page only shows real years
        state_filter=state_filter,
        year_filter=year_filter,
    )

@app.route('/about')
def about():
    return render_template('about.html',
        model_r2=round(metrics['r2'] * 100, 1),
        model_mae=round(metrics['mae'], 1),
        total_records=f"{len(df):,}",
        num_states=len(STATES),
        crime_types=len(CRIME_COLS),
    )

# ── PDF REPORT ────────────────────────────────────────────────
@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    data       = request.json
    state      = data.get('state', 'ALL')
    district   = data.get('district', 'ALL')
    crime_type = data.get('crime_type', 'TOTAL IPC CRIMES')
    year       = int(data.get('year', 2013))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             rightMargin=0.75*inch, leftMargin=0.75*inch,
                             topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles    = getSampleStyleSheet()
    title_sty = ParagraphStyle('T', parent=styles['Title'],
                                fontSize=20, spaceAfter=6,
                                textColor=colors.HexColor('#1e3a8a'))
    h2        = ParagraphStyle('H2', parent=styles['Heading2'],
                                fontSize=13, spaceAfter=4,
                                textColor=colors.HexColor('#1e40af'))
    body      = styles['Normal']

    def make_table(rows, col_widths):
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0),  colors.HexColor('#1e3a8a')),
            ('TEXTCOLOR',     (0,0), (-1,0),  colors.white),
            ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.HexColor('#f0f4ff'), colors.white]),
            ('GRID',          (0,0), (-1,-1),  0.5, colors.HexColor('#cbd5e1')),
            ('FONTSIZE',      (0,0), (-1,-1),  10),
            ('PADDING',       (0,0), (-1,-1),  6),
        ]))
        return t

    story = [
        Paragraph("CrimeWatch Analytics — Prediction Report", title_sty),
        Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}", body),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1e3a8a')),
        Spacer(1, 0.15*inch),
        Paragraph("Query Parameters", h2),
        make_table([['Field','Value'],['State',state],['District',district],
                    ['Crime Type',crime_type],['Year',str(year)]],
                   [2*inch, 4*inch]),
        Spacer(1, 0.2*inch),
    ]

    if state != 'ALL' and district != 'ALL':
        count    = predict_count(year, state, district, crime_type)
        label, _ = risk_label(count)
        forecast = forecast_series(state, district, crime_type)
        summary  = generate_ai_summary(state, district, crime_type, year, count, forecast)

        story += [
            Paragraph("Prediction Result", h2),
            make_table([['Metric','Value'],
                        ['Predicted Cases', str(count) if count is not None else 'N/A'],
                        ['Risk Level', label],
                        ['Model R² Score', f"{round(metrics['r2']*100,1)}%"],
                        ['Mean Absolute Error', str(round(metrics['mae'],1))]],
                       [2.5*inch, 3.5*inch]),
            Spacer(1, 0.2*inch),
            Paragraph("5-Year Forecast (2026–2030)", h2),
            make_table([['Year','Predicted Cases']] +
                       [[str(yr), str(forecast.get(yr,'N/A'))] for yr in range(2026,2031)],
                       [2*inch, 4*inch]),
            Spacer(1, 0.2*inch),
            Paragraph("AI Analysis Summary", h2),
        ]
        for line in summary.split('\n'):
            if line.strip():
                story += [Paragraph(line, body), Spacer(1, 0.05*inch)]

    # Top states table
    top_states = df.groupby('STATE/UT')['TOTAL IPC CRIMES'].sum().nlargest(10).reset_index()
    story += [
        Spacer(1, 0.2*inch),
        Paragraph("Top States by Total IPC Crimes (2001–2013)", h2),
        make_table([['State','Total IPC Crimes']] +
                   [[r['STATE/UT'], f"{int(r['TOTAL IPC CRIMES']):,}"]
                    for _, r in top_states.iterrows()],
                   [3.5*inch, 2.5*inch]),
        Spacer(1, 0.3*inch),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')),
        Paragraph("© CrimeWatch Analytics | NCRB India 2001–2013 | XGBoost Regressor", body),
    ]

    doc.build(story)
    buf.seek(0)
    return send_file(buf, mimetype='application/pdf',
                     download_name=f'crime_report_{state}_{year}.pdf',
                     as_attachment=True)

# ── ENTRY POINT ───────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
