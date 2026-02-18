from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
import subprocess
import sqlite3

app = Flask(__name__)

APP_VERSION = "1.1.0"

# =========================
# MODEL LOADING
# =========================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "demo_model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        subprocess.run(["python3", "train_model.py"], check=True)
    return joblib.load(MODEL_PATH)

model = load_model()

# =========================
# DATABASE LOGGING
# =========================

DB_PATH = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            age REAL,
            bmi REAL,
            sys_bp REAL,
            dia_bp REAL,
            heart_rate REAL,
            exercise REAL,
            prediction TEXT,
            probability REAL
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(age, bmi, sys_bp, dia_bp, heart_rate, exercise, prediction, probability):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO logs(age,bmi,sys_bp,dia_bp,heart_rate,exercise,prediction,probability)
        VALUES(?,?,?,?,?,?,?,?)
    """, (age, bmi, sys_bp, dia_bp, heart_rate, exercise, prediction, probability))
    conn.commit()
    conn.close()

init_db()

# =========================
# MAIN WEB ROUTE
# =========================

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    explanation = None
    error = None

    if request.method == "POST":
        try:
            age = float(request.form["age"])
            bmi = float(request.form["bmi"])
            sys_bp = float(request.form["sys_bp"])
            dia_bp = float(request.form["dia_bp"])
            heart_rate = float(request.form["heart_rate"])
            exercise = float(request.form["exercise"])

            # =========================
            # VALIDATION
            # =========================

            if not (1 <= age <= 120):
                error = "Age must be between 1 and 120."
            elif not (10 <= bmi <= 70):
                error = "BMI must be between 10 and 70."
            elif not (0 <= exercise <= 2):
                error = "Exercise must be 0, 1, or 2."
            elif not (70 <= sys_bp <= 250):
                error = "Systolic BP must be between 70 and 250."
            elif not (40 <= dia_bp <= 150):
                error = "Diastolic BP must be between 40 and 150."
            elif not (30 <= heart_rate <= 220):
                error = "Heart rate must be between 30 and 220."

            if error:
                return render_template(
                    "index.html",
                    prediction=None,
                    probability=None,
                    explanation=None,
                    error=error,
                    version=APP_VERSION)

            # =========================
            # MODEL PREDICTION
            # =========================

            data = np.array([[age, bmi, sys_bp, dia_bp, heart_rate, exercise]])

            prediction_value = int(model.predict(data)[0])
            prob = model.predict_proba(data)[0]
            probability = round(float(prob[1]) * 100, 2)

            prediction = "High Risk" if prediction_value == 1 else "Low Risk"

            # =========================
            # EXPLANATION ENGINE
            # =========================

            reasons = []

            if sys_bp >= 140 or dia_bp >= 90:
                reasons.append("Blood pressure is elevated.")

            if bmi >= 30:
                reasons.append("BMI is in the obese range.")

            if heart_rate >= 100:
                reasons.append("Resting heart rate is high.")

            if exercise == 0:
                reasons.append("Low exercise level increases risk.")

            explanation = " ".join(reasons) if reasons else "No major risk flags detected."

            # =========================
            # LOG TO DATABASE
            # =========================

            log_prediction(age, bmi, sys_bp, dia_bp, heart_rate, exercise,
                           prediction, probability)

        except Exception as e:
            error = "Invalid input. Please check your values."

    return render_template("index.html",
                           prediction=prediction,
                           probability=probability,
                           explanation=explanation,
                           explanation=explanation_text,
                           error=error,
                           version=APP_VERSION)

# =========================
# JSON API ENDPOINT
# =========================

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        payload = request.get_json(force=True)

        age = float(payload["age"])
        bmi = float(payload["bmi"])
        sys_bp = float(payload["sys_bp"])
        dia_bp = float(payload["dia_bp"])
        heart_rate = float(payload["heart_rate"])
        exercise = float(payload["exercise"])

        data = np.array([[age, bmi, sys_bp, dia_bp, heart_rate, exercise]])

        prediction_value = int(model.predict(data)[0])
        prob = model.predict_proba(data)[0]
        probability = round(float(prob[1]) * 100, 2)

        prediction = "High Risk" if prediction_value == 1 else "Low Risk"
        # Simple explanation logic
        explanation = []

if sys_bp > 140:
    explanation.append("Elevated systolic blood pressure")

if dia_bp > 90:
    explanation.append("Elevated diastolic blood pressure")

if bmi > 30:
    explanation.append("High BMI")

if heart_rate > 100:
    explanation.append("Elevated heart rate")

if not explanation:
    explanation_text = "Risk appears influenced by combined moderate factors."
else:
    explanation_text = ", ".join(explanation)


        return jsonify({
            "prediction": prediction,
            "probability_high_risk": probability,
            "version": APP_VERSION})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    y({"error": "Invalid JSON payload"}), 400
