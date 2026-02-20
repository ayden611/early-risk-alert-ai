import os
from datetime import datetime, timezone

import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


app = Flask(__name__)

# ----------------------------
# Database (Postgres on Render)
# ----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Render sometimes gives postgres://... but SQLAlchemy wants postgresql://...
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set. Add it in Render Environment Variables.")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


def init_db():
    """Create table if it doesn't exist."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        age DOUBLE PRECISION NOT NULL,
        bmi DOUBLE PRECISION NOT NULL,
        exercise INTEGER NOT NULL,
        sys_bp DOUBLE PRECISION NOT NULL,
        dia_bp DOUBLE PRECISION NOT NULL,
        heart_rate DOUBLE PRECISION NOT NULL,
        risk_class TEXT NOT NULL,
        probability DOUBLE PRECISION NOT NULL
    );
    """
    with engine.begin() as conn:
        conn.execute(text(create_sql))


init_db()

# ----------------------------
# Model loading
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "demo_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model at {MODEL_PATH}: {e}")


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def to_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def predict_risk(age, bmi, exercise, sys_bp, dia_bp, heart_rate):
    """
    Feature order MUST match training:
    [age, bmi, exercise, sys_bp, dia_bp, heart_rate]
    """
    X = np.array([[age, bmi, exercise, sys_bp, dia_bp, heart_rate]], dtype=float)

    probability = None

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # probability of class 1 (high risk) assumed in column 1
        probability = float(proba[0][1])
    else:
        # fallback: use predict and make probability 0 or 1
        pred = model.predict(X)
        probability = float(pred[0])

    risk_class = "high" if probability >= 0.5 else "low"
    return risk_class, round(probability * 100.0, 1)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    risk_class = None
    probability = None

    # Keep form values after submit
    form_vals = {
        "age": "",
        "bmi": "",
        "exercise": "",
        "sys_bp": "",
        "dia_bp": "",
        "heart_rate": ""
    }

    if request.method == "POST":
        age = to_float(request.form.get("age"))
        bmi = to_float(request.form.get("bmi"))
        exercise = to_int(request.form.get("exercise"))
        sys_bp = to_float(request.form.get("sys_bp"))
        dia_bp = to_float(request.form.get("dia_bp"))
        heart_rate = to_float(request.form.get("heart_rate"))

        form_vals = {
            "age": age,
            "bmi": bmi,
            "exercise": exercise,
            "sys_bp": sys_bp,
            "dia_bp": dia_bp,
            "heart_rate": heart_rate
        }

        risk_class, probability = predict_risk(age, bmi, exercise, sys_bp, dia_bp, heart_rate)
        prediction = True

        # Save to DB
        insert_sql = """
        INSERT INTO predictions (age, bmi, exercise, sys_bp, dia_bp, heart_rate, risk_class, probability)
        VALUES (:age, :bmi, :exercise, :sys_bp, :dia_bp, :heart_rate, :risk_class, :probability);
        """

        db = SessionLocal()
        try:
            db.execute(
                text(insert_sql),
                {
                    "age": age,
                    "bmi": bmi,
                    "exercise": exercise,
                    "sys_bp": sys_bp,
                    "dia_bp": dia_bp,
                    "heart_rate": heart_rate,
                    "risk_class": risk_class,
                    "probability": probability
                },
            )
            db.commit()
        finally:
            db.close()

    return render_template(
        "index.html",
        prediction=prediction,
        risk_class=risk_class,
        probability=probability,
        form_vals=form_vals
    )


@app.route("/history", methods=["GET"])
def history():
    db = SessionLocal()
    try:
        rows = db.execute(
            text("""
                SELECT created_at, age, bmi, exercise, sys_bp, dia_bp, heart_rate, risk_class, probability
                FROM predictions
                ORDER BY created_at DESC
                LIMIT 30;
            """)
        ).fetchall()
    finally:
        db.close()

    # Convert to dictionaries for template readability
    items = []
    for r in rows:
        items.append({
            "created_at": r[0],
            "age": r[1],
            "bmi": r[2],
            "exercise": r[3],
            "sys_bp": r[4],
            "dia_bp": r[5],
            "heart_rate": r[6],
            "risk_class": r[7],
            "probability": r[8],
        })

    return render_template("history.html", items=items)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}
