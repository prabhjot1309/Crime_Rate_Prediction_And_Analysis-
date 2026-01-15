from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import numpy as np
import random

# ==============================================================
# 🔹 GLOBAL STABILITY SETTINGS
# ==============================================================
np.random.seed(0)
random.seed(0)

app = Flask(__name__)

# ==============================================================
# 🔹 CONSTANTS & PATHS
# ==============================================================
API_KEY = "your_secure_api_key_here"  # Replace with a proper secure key

# Use persistent storage instead of /tmp (which resets on Hugging Face)
MODEL_DIR = "model_files"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILE = os.path.join(MODEL_DIR, "crime_model.pkl")
ENCODERS_FILE = os.path.join(MODEL_DIR, "crime_encoders.pkl")

# ==============================================================
# 🔹 LOAD DATA
# ==============================================================
data = pd.read_csv("crime.csv")
data.fillna(0, inplace=True)

crime_columns = data.columns[3:-1]
long_df = data.melt(
    id_vars=["DISTRICT", "YEAR"],
    value_vars=crime_columns,
    var_name="CRIME_TYPE",
    value_name="COUNT"
)
long_df = long_df[long_df["COUNT"] > 0]

# ==============================================================
# 🔹 TRAIN OR LOAD MODEL
# ==============================================================
if os.path.exists(MODEL_FILE) and os.path.exists(ENCODERS_FILE):
    model = joblib.load(MODEL_FILE)
    label_encoders = joblib.load(ENCODERS_FILE)
else:
    district_encoder = LabelEncoder()
    crime_encoder = LabelEncoder()

    long_df["DISTRICT_ENC"] = district_encoder.fit_transform(long_df["DISTRICT"])
    long_df["CRIME_ENC"] = crime_encoder.fit_transform(long_df["CRIME_TYPE"])

    X = long_df[["YEAR", "DISTRICT_ENC"]]
    y = long_df["CRIME_ENC"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, shuffle=True
    )

    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)

    label_encoders = {
        "DISTRICT": district_encoder,
        "CRIME_HEAD": crime_encoder
    }

    joblib.dump(model, MODEL_FILE)
    joblib.dump(label_encoders, ENCODERS_FILE)

# ==============================================================
# 🔹 UTILITY FUNCTION FOR RISK CALCULATION
# ==============================================================
def calculate_risk(year, district, crime_type):
    """Predict crime risk as a stable percentage"""
    try:
        district_enc = label_encoders["DISTRICT"].transform([district])[0]
        crime_enc = label_encoders["CRIME_HEAD"].transform([crime_type])[0]
        X_input = [[year, district_enc]]

        # Get model prediction probability
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0].max()
        else:
            proba = 0.5  # fallback if classifier doesn’t support proba

        risk = round(float(proba) * 100, 2)
        return min(max(risk, 0), 100)
    except Exception as e:
        print("Risk calculation error:", e)
        return 0

# ==============================================================
# 🔹 ROUTES
# ==============================================================
@app.route('/')
def home():
    years = sorted(data['YEAR'].unique())
    cities = sorted(data['DISTRICT'].unique())
    crime_types = sorted(long_df['CRIME_TYPE'].unique())
    return render_template("index.html", years=years, cities=cities, crime_types=crime_types)


@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    if request.method == 'POST':
        try:
            year = int(request.form['year'])
            district = request.form['city']
            crime_type = request.form['crime_type']

            risk = calculate_risk(year, district, crime_type)
            return render_template('predictions.html', prediction=f"{risk}% Risk")
        except Exception as e:
            return f"Error: {e}"
    return render_template('predictions.html', prediction=None)


@app.route('/reports')
def reports():
    return render_template("reports.html")


@app.route('/data')
def data_page():
    return render_template("data.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/result', methods=['POST'])
def result():
    selected_year = int(request.form['year'])
    selected_city = request.form['city']

    filtered = data[(data['YEAR'] == selected_year) & (data['DISTRICT'] == selected_city)]

    if filtered.empty:
        return render_template("result.html", result=f"No data found for {selected_city} in {selected_year}.")

    total_crimes = int(filtered['TOTAL IPC CRIMES'].sum())
    result_data = {
        'city': selected_city,
        'year': selected_year,
        'crimes': total_crimes
    }

    return render_template("result.html", result=result_data)

# ==============================================================
# 🔹 MAIN ENTRY POINT
# ==============================================================
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    app.run(host='0.0.0.0', port=7860)
