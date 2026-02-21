import os
import math
from datetime import datetime, date

from flask import Flask, render_template, request, redirect, url_for, flash
from sqlalchemy import create_engine, text
import joblib


# ============================================================
# APP SETUP
# ============================================================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")


# ============================================================
# DATABASE SETUP
# ============================================================
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

engine = None
if DATABASE_URL:
    # Render Postgres often provides postgres:// which SQLAlchemy wants as postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
else:
    # Local/dev fallback if DATABASE_URL is missing
    engine = create_engine("sqlite:///local.db", connect_args={"check_same_thread": False})


def init_db():
    """Create table if it doesn't exist."""
    if not engine:
        return

    # Works on Postgres. For sqlite, SERIAL isn't valid, but sqlite ignores type-ish.
    # If you want perfect sqlite support, we can add sqlite-specific DDL later.
    ddl = """
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
        probability FLOAT,
        confidence FLOAT
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


init_db()


# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = os.environ.get("MODEL_PATH", "demo_model.pkl")
model = None
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None


# ============================================================
# HELPERS
# ============================================================
def to_float(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def sigmoid(z):
    # safe-ish sigmoid
    if z is None:
        return 0.5
    z = max(-60.0, min(60.0, float(z)))
    return 1.0 / (1.0 + math.exp(-z))


def normalize_exercise(ex):
    """
    Accepts: low / moderate / high (case-insensitive)
    Stores the text version in DB, but also returns numeric encoding for model if needed.
    """
    if not ex:
        return ("low", 0.0)
    ex_clean = str(ex).strip().lower()
    if ex_clean in ("low", "l"):
        return ("low", 0.0)
    if ex_clean in ("moderate", "medium", "mid", "m"):
        return ("moderate", 1.0)
    if ex_clean in ("high", "h"):
        return ("high", 2.0)
    # default fallback
    return (ex_clean, 0.0)


def format_pct(p):
    """
    p is expected 0..1 float.
    Returns a string like '23.4%' or '<0.1%' (so low risk doesn't look like 0.0%).
    """
    if p is None:
        return "—"
    p = clamp(float(p), 0.0, 1.0)
    pct = p * 100.0
    if 0.0 < pct < 0.1:
        return "<0.1%"
    return f"{pct:.1f}%"


def predict_probability(features_dict):
    """
    Robust inference:
    - If model has predict_proba -> use it
    - Else decision_function -> sigmoid
    - Else predict -> 0/1
    """
    if not model:
        return 0.5

    # Many sklearn pipelines accept dict-like via DataFrame.
    # Without pandas dependency, try array fallback.
    # We'll try dict->list in a stable order that matches our app fields.
    feature_order = ["age", "bmi", "exercise_level_num", "systolic_bp", "diastolic_bp", "heart_rate"]
    X_row = [[features_dict.get(k) for k in feature_order]]

    # Try predict_proba
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_row)[0][1]
            return clamp(float(proba), 0.0, 1.0)
        except Exception:
            pass

    # Try decision_function
    if hasattr(model, "decision_function"):
        try:
            score = model.decision_function(X_row)
            if isinstance(score, (list, tuple)):
                score = score[0]
            return clamp(sigmoid(score), 0.0, 1.0)
        except Exception:
            pass

    # Fallback to predict
    try:
        pred = model.predict(X_row)[0]
        return 1.0 if int(pred) == 1 else 0.0
    except Exception:
        return 0.5


def compute_result(age, bmi, exercise_level_text, exercise_level_num, systolic_bp, diastolic_bp, heart_rate):
    features = {
        "age": age,
        "bmi": bmi,
        "exercise_level_num": exercise_level_num,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "heart_rate": heart_rate,
    }

    proba = predict_probability(features)

    # Risk threshold (can tune later)
    risk_class = "high" if proba >= 0.5 else "low"
    risk_label = "High Risk" if risk_class == "high" else "Low Risk"

    # Confidence: distance from 0.5 scaled to 0..1
    confidence = clamp(abs(proba - 0.5) * 2.0, 0.0, 1.0)

    return {
        "risk_label": risk_label,
        "risk_class": risk_class,
        "probability": proba,          # raw 0..1
        "confidence": confidence,      # raw 0..1
        "probability_str": format_pct(proba),
        "confidence_str": format_pct(confidence),
    }


# ============================================================
# ROUTES
# ============================================================
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    # default form values
    form = {
        "age": "",
        "bmi": "",
        "exercise_level": "low",
        "systolic_bp": "",
        "diastolic_bp": "",
        "heart_rate": "",
        "ack": False,
    }

    if request.method == "POST":
        form["age"] = request.form.get("age", "").strip()
        form["bmi"] = request.form.get("bmi", "").strip()
        form["exercise_level"] = request.form.get("exercise_level", "low").strip()
        form["systolic_bp"] = request.form.get("systolic_bp", "").strip()
        form["diastolic_bp"] = request.form.get("diastolic_bp", "").strip()
        form["heart_rate"] = request.form.get("heart_rate", "").strip()
        form["ack"] = bool(request.form.get("ack"))

        try:
            age = to_float(form["age"])
            bmi = to_float(form["bmi"])
            systolic_bp = to_float(form["systolic_bp"])
            diastolic_bp = to_float(form["diastolic_bp"])
            heart_rate = to_float(form["heart_rate"])

            ex_text, ex_num = normalize_exercise(form["exercise_level"])

            # Basic validation
            if not form["ack"]:
                raise ValueError("Please check the acknowledgement box.")
            if None in (age, bmi, systolic_bp, diastolic_bp, heart_rate):
                raise ValueError("Missing one or more values.")
            if age <= 0 or bmi <= 0 or systolic_bp <= 0 or diastolic_bp <= 0 or heart_rate <= 0:
                raise ValueError("Values must be greater than 0.")

            result = compute_result(age, bmi, ex_text, ex_num, systolic_bp, diastolic_bp, heart_rate)

            # Save to DB
            if engine:
                with engine.begin() as conn:
                    conn.execute(
                        text("""
                            INSERT INTO predictions
                            (age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate,
                             risk_label, risk_class, probability, confidence)
                            VALUES
                            (:age, :bmi, :exercise_level, :systolic_bp, :diastolic_bp, :heart_rate,
                             :risk_label, :risk_class, :probability, :confidence)
                        """),
                        {
                            "age": age,
                            "bmi": bmi,
                            "exercise_level": ex_text,
                            "systolic_bp": systolic_bp,
                            "diastolic_bp": diastolic_bp,
                            "heart_rate": heart_rate,
                            "risk_label": result["risk_label"],
                            "risk_class": result["risk_class"],
                            "probability": float(result["probability"]),
                            "confidence": float(result["confidence"]),
                        },
                    )

        except Exception:
            error = "Oops: Invalid input. Please check your values."

    return render_template("index.html", result=result, error=error, form=form)


@app.route("/history", methods=["GET"])
def history():
    if not engine:
        return render_template("history.html", history=[], filters={})

    # Filters from query params
    risk_class = request.args.get("risk_class", "").strip().lower()  # low/high
    min_prob = request.args.get("min_prob", "").strip()              # percent like 10
    max_prob = request.args.get("max_prob", "").strip()              # percent like 50
    date_from = request.args.get("date_from", "").strip()            # YYYY-MM-DD
    date_to = request.args.get("date_to", "").strip()                # YYYY-MM-DD

    where = []
    params = {}

    if risk_class in ("low", "high"):
        where.append("risk_class = :risk_class")
        params["risk_class"] = risk_class

    min_prob_f = to_float(min_prob)
    if min_prob_f is not None:
        where.append("probability >= :min_p")
        params["min_p"] = clamp(min_prob_f / 100.0, 0.0, 1.0)

    max_prob
