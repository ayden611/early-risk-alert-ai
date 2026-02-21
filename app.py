import os
from datetime import datetime

import joblib
from flask import Flask, render_template, request, redirect, url_for, flash
from sqlalchemy import create_engine, text

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# ----------------------------
# DATABASE
# ----------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")

# Render sometimes provides postgres:// which SQLAlchemy wants as postgresql://
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None


def init_db():
    if not engine:
        return
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                age FLOAT,
                bmi FLOAT,
                exercise_level TEXT,
                systolic_bp FLOAT,
                diastolic_bp FLOAT,
                heart_rate FLOAT,
                risk_label TEXT,
                risk_class TEXT,
                probability FLOAT,      -- STORE AS 0..1 (NOT percent)
                confidence FLOAT        -- STORE AS 0..1 (NOT percent)
            );
        """))


init_db()

# ----------------------------
# LOAD MODEL
# ----------------------------
model = joblib.load("demo_model.pkl")


def clamp01(x):
    try:
        x = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, x))


# ----------------------------
# HOME ROUTE
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    # default form state
    form = {
        "age": "",
        "bmi": "",
        "exercise_level": "Low",
        "systolic_bp": "",
        "diastolic_bp": "",
        "heart_rate": "",
        "ack": False
    }

    if request.method == "POST":
        try:
            form["age"] = request.form.get("age", "")
            form["bmi"] = request.form.get("bmi", "")
            form["exercise_level"] = request.form.get("exercise_level", "Low")
            form["systolic_bp"] = request.form.get("systolic_bp", "")
            form["diastolic_bp"] = request.form.get("diastolic_bp", "")
            form["heart_rate"] = request.form.get("heart_rate", "")
            form["ack"] = (request.form.get("ack") == "on")

            if not form["ack"]:
                raise ValueError("You must acknowledge the disclaimer checkbox.")

            age = float(form["age"])
            bmi = float(form["bmi"])
            systolic_bp = float(form["systolic_bp"])
            diastolic_bp = float(form["diastolic_bp"])
            heart_rate = float(form["heart_rate"])

            # simple mapping (keep consistent with training)
            exercise_map = {"Low": 0, "Moderate": 1, "High": 2}
            exercise_val = exercise_map.get(form["exercise_level"], 0)

            X = [[age, bmi, exercise_val, systolic_bp, diastolic_bp, heart_rate]]

            # Predict
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(X)[0][1])  # probability of "high risk"
            else:
                # fallback: treat prediction as 0/1 probability
                pred = int(model.predict(X)[0])
                proba = float(pred)

            proba = clamp01(proba)

            # Choose class label from probability threshold
            # (You can tune this later; 0.50 is default)
            risk_class = "high" if proba >= 0.50 else "low"
            risk_label = "High Risk" if risk_class == "high" else "Low Risk"

            # confidence: distance from threshold (0.5) scaled
            confidence = clamp01(abs(proba - 0.5) * 2)

            result = {
                "risk_label": risk_label,
                "risk_class": risk_class,
                "probability": proba,     # 0..1
                "confidence": confidence  # 0..1
            }

            # Save to DB
            if engine:
                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO predictions (
                            age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate,
                            risk_label, risk_class, probability, confidence
                        ) VALUES (
                            :age, :bmi, :exercise_level, :systolic_bp, :diastolic_bp, :heart_rate,
                            :risk_label, :risk_class, :probability, :confidence
                        )
                    """), {
                        "age": age,
                        "bmi": bmi,
                        "exercise_level": form["exercise_level"],
                        "systolic_bp": systolic_bp,
                        "diastolic_bp": diastolic_bp,
                        "heart_rate": heart_rate,
                        "risk_label": risk_label,
                        "risk_class": risk_class,
                        "probability": proba,
                        "confidence": confidence
                    })

        except Exception:
            error = "Oops: Invalid input. Please check your values."

    return render_template("index.html", result=result, error=error, form=form)


# ----------------------------
# HISTORY (WITH FILTERING)
# ----------------------------
@app.route("/history", methods=["GET"])
def history():
    if not engine:
        return render_template("history.html", history=[], filters={})

    # Filters from query params
    risk_class = request.args.get("risk_class", "").strip()  # low/high
    min_prob = request.args.get("min_prob", "").strip()      # percent like 10, 50
    max_prob = request.args.get("max_prob", "").strip()
    date_from = request.args.get("date_from", "").strip()    # YYYY-MM-DD
    date_to = request.args.get("date_to", "").strip()

    where = []
    params = {}

    if risk_class in ("low", "high"):
        where.append("risk_class = :risk_class")
        params["risk_class"] = risk_class
    def pct_to_
