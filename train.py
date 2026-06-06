"""
run once: python3 train.py
produces model_files/ with xgb_model.pkl, encoders.pkl, metrics.pkl, analytics.json
"""
import pandas as pd
import numpy as np
import joblib, json, os, sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor

print(">>> train.py started", flush=True)

np.random.seed(42)
BASE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE, "model_files"), exist_ok=True)

print(">>> Loading CSV...", flush=True)
df = pd.read_csv(os.path.join(BASE, "crime.csv"))
df.fillna(0, inplace=True)
df['STATE/UT'] = df['STATE/UT'].str.upper().str.strip()
df['DISTRICT'] = df['DISTRICT'].str.upper().str.strip()
df = df[df['DISTRICT'] != 'TOTAL'].copy()
print(f">>> Loaded {len(df)} rows", flush=True)

CRIME_COLS = [c for c in df.columns
              if c not in ['STATE/UT', 'DISTRICT', 'YEAR', 'TOTAL IPC CRIMES']]

long_df = df.melt(id_vars=['STATE/UT', 'DISTRICT', 'YEAR'],
                  value_vars=CRIME_COLS,
                  var_name='CRIME_TYPE', value_name='COUNT')

print(">>> Encoding labels...", flush=True)
state_enc    = LabelEncoder()
district_enc = LabelEncoder()
crime_enc    = LabelEncoder()

long_df['STATE_ENC']    = state_enc.fit_transform(long_df['STATE/UT'])
long_df['DISTRICT_ENC'] = district_enc.fit_transform(long_df['DISTRICT'])
long_df['CRIME_ENC']    = crime_enc.fit_transform(long_df['CRIME_TYPE'])

X = long_df[['YEAR', 'STATE_ENC', 'DISTRICT_ENC', 'CRIME_ENC']]
y = long_df['COUNT']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(">>> Training XGBoost (fast config)...", flush=True)
model = XGBRegressor(
    n_estimators=100,      # reduced from 200 — still accurate, 2x faster
    max_depth=5,           # reduced from 6
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=1,              # FIXED: was -1, causes hangs in Docker
    tree_method='hist',    # fastest CPU method
    verbosity=0,
)
model.fit(X_train, y_train)
print(">>> Training done", flush=True)

preds = model.predict(X_test)
mae  = mean_absolute_error(y_test, preds)
r2   = r2_score(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f">>> MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}", flush=True)

print(">>> Saving model files...", flush=True)
joblib.dump(model,
    os.path.join(BASE, 'model_files', 'xgb_model.pkl'))
joblib.dump({'state': state_enc, 'district': district_enc, 'crime': crime_enc},
    os.path.join(BASE, 'model_files', 'encoders.pkl'))
joblib.dump({'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)},
    os.path.join(BASE, 'model_files', 'metrics.pkl'))

print(">>> Building analytics.json...", flush=True)
year_trend  = df.groupby('YEAR')['TOTAL IPC CRIMES'].sum().reset_index()
state_total = df.groupby('STATE/UT')['TOTAL IPC CRIMES'].sum().nlargest(10).reset_index()
crime_totals = {c: int(df[c].sum()) for c in CRIME_COLS}
top_crimes   = sorted(crime_totals.items(), key=lambda x: x[1], reverse=True)[:10]
districts_by_state = (df.groupby('STATE/UT')['DISTRICT']
                        .unique()
                        .apply(lambda x: sorted(x.tolist()))
                        .to_dict())

analytics = {
    'year_trend': {
        'years':  year_trend['YEAR'].tolist(),
        'totals': year_trend['TOTAL IPC CRIMES'].tolist()
    },
    'state_data': {
        'states': state_total['STATE/UT'].tolist(),
        'totals': state_total['TOTAL IPC CRIMES'].tolist()
    },
    'top_crimes': [{'crime': c, 'count': n} for c, n in top_crimes],
    'crime_cols': CRIME_COLS,
    'states':  sorted(df['STATE/UT'].unique().tolist()),
    'years':   sorted(df['YEAR'].unique().tolist()),
    'districts_by_state': districts_by_state,
}
with open(os.path.join(BASE, 'model_files', 'analytics.json'), 'w') as f:
    json.dump(analytics, f)

print(">>> All files saved to model_files/ — training complete!", flush=True)
