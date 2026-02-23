import os
from datetime import datetime

import numpy as np
import joblib

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

# ----------------------------
# App Setup
# ----------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev")

db_url = os.getenv("DATABASE_URL", "")

# Render sometimes provides postgres:// (SQLAlchemy wants postgresql://)
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

# Local fallback
if not db_url:
    db_url = "sqlite:///local.db"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ----------------------------
# Load Model
# ----------------------------
MODEL_PATH = "demo_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# ----------------------------
# Database Model
# ----------------------------
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.String(32), nullable=False)

    systolic_bp = db.Column(db.Float, nullable=False)
    diastolic_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    risk_label = db.Column(db.String(32), nullable=False)
    prob_high = db.Column(db.Float, nullable=False)


with app.app_context():
    db.create_all()

# ----------------------------
# Helpers
# ----------------------------
def normalize_exercise(x: str) -> str:
    x = (x or "").strip().lower()
    if x in ["0", "low"]:
        return "Low"
    if x in ["1", "moderate", "medium"]:
        return "Moderate"
    if x in ["2", "high"]:
        return "High"
    # default
    return "Low"


def exercise_to_numeric(x: str) -> float:
    x = (x or "").strip().lower()
    if x in ["moderate", "medium", "1"]:
        return 1.0
    if x in ["high", "2"]:
        return 2.0
    return 0.0


def predict_risk(age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate):
    # If model is missing, provide a safe fallback
    if model is None:
        # simple heuristic fallback so app never breaks
        score = 0
        if age >= 50: score += 1
        if bmi >= 30: score += 1
        if systolic_bp >= 140 or diastolic_bp >= 90: score += 1
        if heart_rate >= 95: score += 1
        prob_high = min(0.15 + 0.2 * score, 0.95)
        label = "High Risk" if prob_high >= 0.5 else "Low Risk"
        return label, float(prob_high)

    ex_num = exercise_to_numeric(exercise_level)

    X = np.array([[age, bmi, ex_num, systolic_bp, diastolic_bp, heart_rate]], dtype=float)

    pred = int(model.predict(X)[0])
    prob_high = float(model.predict_proba(X)[0][1])

    label = "High Risk" if pred == 1 else "Low Risk"
    return label, prob_high


def save_prediction(age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate, label, prob_high):
    row = Prediction(
        age=age,
        bmi=bmi,
        exercise_level=exercise_level,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        heart_rate=heart_rate,
        risk_label=label,
        prob_high=prob_high,
    )
    db.session.add(row)
    db.session.commit()
    return row


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return jsonify(status="ok")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            age = float(request.form.get("age", "0"))
            bmi = float(request.form.get("bmi", "0"))
            exercise_level = normalize_exercise(request.form.get("exercise_level", "Low"))
            systolic_bp = float(request.form.get("systolic_bp", "0"))
            diastolic_bp = float(request.form.get("diastolic_bp", "0"))
            heart_rate = float(request.form.get("heart_rate", "0"))

            label, prob_high = predict_risk(
                age=age,
                bmi=bmi,
                exercise_level=exercise_level,
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                heart_rate=heart_rate,
            )

            # Save to DB (server submit MUST save)
            save_prediction(age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate, label, prob_high)

            prediction = label
            probability = round(prob_high * 100, 2)

        except Exception as e:
            db.session.rollback()
            error = f"{type(e).__name__}: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error,
    )


# Instant JS route (returns JSON, also saves to DB)
@app.post("/api/predict")
def api_predict():
    try:
        data = request.get_json(force=True) or {}

        age = float(data.get("age", 0))
        bmi = float(data.get("bmi", 0))
        exercise_level = normalize_exercise(data.get("exercise_level", "Low"))
        systolic_bp = float(data.get("systolic_bp", 0))
        diastolic_bp = float(data.get("diastolic_bp", 0))
        heart_rate = float(data.get("heart_rate", 0))

        label, prob_high = predict_risk(
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
        )

        save_prediction(age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate, label, prob_high)

        return jsonify(
            risk=label,
            prob_high=round(prob_high * 100, 2),
        )

    except Exception as e:
        db.session.rollback()
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.get("/history")
def history():
    rows = (
        Prediction.query.order_by(Prediction.created_at.desc())
        .limit(50)
        .all()
    )
    return render_template("history.html", rows=rows)


@app.get("/debug")
def debug():
    try:
        count = Prediction.query.count()
        return jsonify(
            status="ok",
            db_uri=app.config["SQLALCHEMY_DATABASE_URI"],
            rows=count,
        )
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


# Local dev only (Render uses gunicorn via wsgi.py)
if __name__ == "__main__":
    app.run(debug=True)
