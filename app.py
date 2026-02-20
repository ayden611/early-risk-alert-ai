from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib
import os
import json
from datetime import datetime

app = Flask(__name__)

# -----------------------------
# Model loading (safe for local + Render)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "demo_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# -----------------------------
# Simple history storage (no DB)
# Render allows writing to /tmp
# -----------------------------
HISTORY_PATH = os.path.join("/tmp", "pred_history.json")


def load_history():
    try:
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def save_history(history):
    try:
        with open(HISTORY_PATH, "w") as f:
            json.dump(history, f)
    except Exception:
        pass


def safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default


def safe_int(val, default=0):
    try:
        return int(float(val))
    except Exception:
        return default


# -----------------------------
# Main route (FIXES 405)
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    risk_class = None

    # Always define form_vals so template never crashes
    form_vals = {
        "age": "",
        "bmi": "",
        "exercise": "",
        "sys_bp": "",
        "dia_bp": "",
        "heart_rate": ""
    }

    if request.method == "POST":
        # Pull values from form
        form_vals["age"] = request.form.get("age", "")
        form_vals["bmi"] = request.form.get("bmi", "")
        form_vals["exercise"] = request.form.get("exercise", "")
        form_vals["sys_bp"] = request.form.get("sys_bp", "")
        form_vals["dia_bp"] = request.form.get("dia_bp", "")
        form_vals["heart_rate"] = request.form.get("heart_rate", "")

        # Convert safely
        age = safe_float(form_vals["age"], 0.0)
        bmi = safe_float(form_vals["bmi"], 0.0)
        exercise = safe_int(form_vals["exercise"], 0)  # expected: 0/1/2
        sys_bp = safe_float(form_vals["sys_bp"], 0.0)
        dia_bp = safe_float(form_vals["dia_bp"], 0.0)
        heart_rate = safe_float(form_vals["heart_rate"], 0.0)

        # Predict
        data = np.array([[age, bmi, exercise, sys_bp, dia_bp, heart_rate]])

        pred = int(model.predict(data)[0])

        # Probability handling (if model supports predict_proba)
        prob_high = None
        if hasattr(model, "predict_proba"):
            try:
                prob_high = float(model.predict_proba(data)[0][1])
            except Exception:
                prob_high = None

        # Output labels
        prediction = "High Risk" if pred == 1 else "Low Risk"

        # If no probability available, fall back to 0 or 100 based on class
        if prob_high is None:
            probability = 100.0 if pred == 1 else 0.0
        else:
            probability = round(prob_high * 100, 2)

        # CSS/visual classification label for template
        risk_class = "high" if pred == 1 else "low"

        # Save to history (latest first)
        history = load_history()
        history.insert(0, {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "inputs": {
                "age": age,
                "bmi": bmi,
                "exercise": exercise,
                "sys_bp": sys_bp,
                "dia_bp": dia_bp,
                "heart_rate": heart_rate
            },
            "prediction": prediction,
            "probability": probability
        })
        history = history[:50]  # keep last 50
        save_history(history)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        risk_class=risk_class,
        form_vals=form_vals
    )


# -----------------------------
# History page (no DB)
# -----------------------------
@app.route("/history", methods=["GET"])
def history():
    history = load_history()
    return render_template("history.html", history=history)


# -----------------------------
# Health check route (optional)
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200
