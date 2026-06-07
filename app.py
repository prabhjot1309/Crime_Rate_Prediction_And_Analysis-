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
from reportlab.lib.enums import TA_CENTER, TA_LEFT

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

model      = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
encoders   = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
metrics    = joblib.load(os.path.join(MODEL_DIR, "metrics.pkl"))

with open(os.path.join(MODEL_DIR, "analytics.json")) as f:
    analytics = json.load(f)

STATES     = analytics['states']
DISTRICTS  = analytics['districts_by_state']
YEARS      = analytics['years']

# ── HELPERS ───────────────────────────────────────────────────
def safe_encode(encoder, value):
    classes = list(encoder.classes_)
    if value in classes:
        return encoder.transform([value])[0]
    return None

def predict_count(year: int, state: str, district: str, crime_type: str):
    s = safe_encode(encoders['state'], state)
    d = safe_encode(encoders['district'], district)
    c = safe_encode(encoders['crime'], crime_type)
    if any(v is None for v in [s, d, c]):
        return None
    val = model.predict([[year, s, d, c]])[0]
    return max(0, round(float(val), 1))

def forecast_series(state: str, district: str, crime_type: str,
                    from_year=2014, to_year=2030):
    """Return {year: predicted_count} using XGBoost extrapolation."""
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
    if count is None:   return "Unknown", "gray"
    if count == 0:      return "None",    "green"
    if count < 50:      return "Low",     "green"
    if count < 200:     return "Medium",  "yellow"
    if count < 500:     return "High",    "orange"
    return "Very High", "red"

def generate_ai_summary(state, district, crime_type, year, count, forecast):
    """Rule-based AI summary (no external API needed)."""
    label, _ = risk_label(count)
    trend_vals = list(forecast.values())
    if len(trend_vals) >= 3:
        recent = trend_vals[:3]
        direction = "rising" if recent[-1] > recent[0] else "declining"
    else:
        direction = "stable"
    
    lines = [
        f"📍 District: {district}, {state}",
        f"📅 Year: {year} | Crime: {crime_type}",
        f"",
        f"🔍 Risk Level: {label}",
        f"📊 Predicted Cases: {count if count is not None else 'N/A'}",
        f"",
        f"📈 Forecast Trend (2014–2020): {direction.upper()}",
    ]
    if direction == "rising":
        lines.append("⚠️  Increasing trend detected. Authorities may consider enhanced patrolling.")
    else:
        lines.append("✅  Declining trend noted. Current measures appear effective.")

    high_states = df.groupby('STATE/UT')[crime_type].sum().nlargest(3).index.tolist() if crime_type in df.columns else []
    if state in high_states:
        lines.append(f"🚨  {state} is among the top 3 states for {crime_type}.")
    
    return "\n".join(lines)

# ── ROUTES ────────────────────────────────────────────────────
@app.route('/')
def index():
    year_trend = analytics['year_trend']
    top_crimes = analytics['top_crimes']
    state_data = analytics['state_data']
    total_crimes = int(df['TOTAL IPC CRIMES'].sum())
    return render_template('index.html',
        states=STATES,
        years=YEARS,
        crime_cols=CRIME_COLS,
        year_trend=json.dumps(year_trend),
        top_crimes=json.dumps(top_crimes),
        state_data=json.dumps(state_data),
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
    data = request.json
    year      = int(data['year'])
    state     = data['state'].upper().strip()
    district  = data['district'].upper().strip()
    crime_type = data['crime_type']

    count = predict_count(year, state, district, crime_type)
    label, color = risk_label(count)
    forecast = forecast_series(state, district, crime_type, from_year=max(year, 2014), to_year=2030)
    summary = generate_ai_summary(state, district, crime_type, year, count, forecast)

    # Historical actuals
    hist = df[(df['STATE/UT'] == state) & (df['DISTRICT'] == district)]
    historical = {}
    if crime_type in df.columns:
        for _, row in hist.iterrows():
            historical[int(row['YEAR'])] = int(row[crime_type])

    return jsonify({
        'count': count,
        'label': label,
        'color': color,
        'summary': summary,
        'forecast': {str(k): v for k, v in forecast.items()},
        'historical': {str(k): v for k, v in historical.items()},
    })

@app.route('/predictions')
def predictions():
    return render_template('predictions.html',
        states=STATES,
        crime_cols=CRIME_COLS,
        years=list(range(2001, 2031)),
    )

@app.route('/reports')
def reports():
    # State summary table
    state_summary = df.groupby('STATE/UT').agg(
        total_crimes=('TOTAL IPC CRIMES', 'sum'),
        years_count=('YEAR', 'nunique'),
        districts=('DISTRICT', 'nunique')
    ).reset_index().sort_values('total_crimes', ascending=False).head(15)
    rows = state_summary.to_dict('records')
    return render_template('reports.html',
        state_rows=rows,
        model_r2=round(metrics['r2'] * 100, 1),
        model_mae=round(metrics['mae'], 1),
        model_rmse=round(metrics['rmse'], 1),
        top_crimes=json.dumps(analytics['top_crimes']),
        state_data=json.dumps(analytics['state_data']),
        year_trend=json.dumps(analytics['year_trend']),
    )

@app.route('/data')
def data_page():
    # Paginated data
    page = int(request.args.get('page', 1))
    per_page = 50
    state_filter = request.args.get('state', '')
    year_filter  = request.args.get('year', '')

    filtered = df.copy()
    if state_filter:
        filtered = filtered[filtered['STATE/UT'] == state_filter.upper()]
    if year_filter:
        filtered = filtered[filtered['YEAR'] == int(year_filter)]

    total = len(filtered)
    start = (page - 1) * per_page
    end   = start + per_page
    page_data = filtered.iloc[start:end][
        ['STATE/UT','DISTRICT','YEAR','MURDER','RAPE','THEFT','ROBBERY','TOTAL IPC CRIMES']
    ].to_dict('records')

    return render_template('data.html',
        rows=page_data,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=(total // per_page) + 1,
        states=STATES,
        years=YEARS,
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
    data = request.json
    state      = data.get('state', 'ALL')
    district   = data.get('district', 'ALL')
    crime_type = data.get('crime_type', 'TOTAL IPC CRIMES')
    year       = int(data.get('year', 2013))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             rightMargin=0.75*inch, leftMargin=0.75*inch,
                             topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title2', parent=styles['Title'],
                                  fontSize=20, spaceAfter=6, textColor=colors.HexColor('#1e3a8a'))
    h2 = ParagraphStyle('H2', parent=styles['Heading2'],
                         fontSize=13, textColor=colors.HexColor('#1e40af'), spaceAfter=4)
    body = styles['Normal']

    story = []
    story.append(Paragraph("CrimeWatch Analytics — Prediction Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}", body))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1e3a8a')))
    story.append(Spacer(1, 0.15*inch))

    # Query details
    story.append(Paragraph("Query Parameters", h2))
    params = [['Field', 'Value'],
              ['State', state], ['District', district],
              ['Crime Type', crime_type], ['Year', str(year)]]
    t = Table(params, colWidths=[2*inch, 4*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a8a')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f0f4ff'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2*inch))

    # Prediction
    if state != 'ALL' and district != 'ALL':
        count = predict_count(year, state, district, crime_type)
        label, _ = risk_label(count)
        forecast = forecast_series(state, district, crime_type)
        summary  = generate_ai_summary(state, district, crime_type, year, count, forecast)

        story.append(Paragraph("Prediction Result", h2))
        pred_data = [['Metric', 'Value'],
                     ['Predicted Cases', str(count) if count is not None else 'N/A'],
                     ['Risk Level', label],
                     ['Model R² Score', f"{round(metrics['r2']*100,1)}%"],
                     ['Mean Absolute Error', str(round(metrics['mae'], 1))]]
        pt = Table(pred_data, colWidths=[2.5*inch, 3.5*inch])
        pt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#fafafa'), colors.white]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(pt)
        story.append(Spacer(1, 0.2*inch))

        # Forecast table
        story.append(Paragraph("5-Year Forecast (2026–2030)", h2))
        fc_rows = [['Year', 'Predicted Cases']]
        for yr in range(2026, 2031):
            fc_rows.append([str(yr), str(forecast.get(yr, 'N/A'))])
        ft = Table(fc_rows, colWidths=[2*inch, 4*inch])
        ft.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a8a')),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f0f4ff'), colors.white]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(ft)
        story.append(Spacer(1, 0.2*inch))

        # AI Summary
        story.append(Paragraph("AI Analysis Summary", h2))
        for line in summary.split('\n'):
            if line.strip():
                story.append(Paragraph(line, body))
                story.append(Spacer(1, 0.05*inch))

    # State overview table
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Top States by Total IPC Crimes (2001–2013)", h2))
    top_states = df.groupby('STATE/UT')['TOTAL IPC CRIMES'].sum().nlargest(10).reset_index()
    tbl = [['State', 'Total IPC Crimes']]
    for _, row in top_states.iterrows():
        tbl.append([row['STATE/UT'], f"{int(row['TOTAL IPC CRIMES']):,}"])
    st_tbl = Table(tbl, colWidths=[3.5*inch, 2.5*inch])
    st_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a8a')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8fafc'), colors.white]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(st_tbl)

    story.append(Spacer(1, 0.3*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
    story.append(Paragraph("© CrimeWatch Analytics | Data: NCRB India 2001–2013 | Model: XGBoost Regressor", body))

    doc.build(story)
    buf.seek(0)
    return send_file(buf, mimetype='application/pdf',
                     download_name=f'crime_report_{state}_{year}.pdf',
                     as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
