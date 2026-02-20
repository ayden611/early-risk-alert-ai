import os
from datetime import datetime

import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from sqlalchemy import create_engine, text

app = Flask(__name__)

# -----------------------------
# Database (Render Postgres)
# -----------------------------
def get_database_url() -> str | None:
    url = os.getenv("DATABASE_URL")
    if not url:
        return None

    # Render often provides postgres://... which SQLAlchemy prefers as postgresql+psycopg2://...
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)

    # Make SSL explicit for hosted DB connections
    if "sslmode=" not in url:
        joiner = "&" if "?" in url else "?"
        url = f"{url}{joiner}sslmode=require"

    return url

DATABASE_URL = get_database_url()
engine = create_engine(DATABASE_URL, pool_pre_ping=True) if DATABASE_URL else None


def init_db():
    """Create table if it doesn't exist."""
    if not engine:
        return
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                age FLOAT NOT NULL,
                bmi FLOAT NOT NULL,
                exercise INTEGER NOT NULL,
                sys_bp FLOAT NOT NULL,
                dia_bp FLOAT NOT NULL,
                heart_rate FLOAT NOT NULL,
                risk_class TEXT NOT NULL,
                probability FLOAT NOT NULL
            );
        """))


# -----------------------------
# Model loading
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "demo_model.pkl")
model = joblib.load(MODEL_PATH)  # will work once scikit-learn is installed


def predict_risk(age, bmi, exercise, sys_bp, dia_bp, heart_rate):
    X = np.array([[age, bmi, exercise, sys_bp, dia_bp, heart_rate]], dtype=float)

    # Try predict_proba first (best)
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])
    else:
        # Fallback: decision_function -> sigmoid-ish scale
        score = float(model.decision_function(X)[0])
        proba = 1.0 / (1.0 + np.exp(-score))

    risk_class = "high" if proba >= 0.5 else "low"
    return risk_class, proba


@app.route("/", methods=["GET", "POST"])
def index():
    init_db()

    form_vals = {
        "age": "",
        "bmi": "",
        "exercise": "",
        "sys_bp": "",
        "dia_bp": "",
        "heart_rate": "",
    }

    prediction = None
    risk_class = None
    probability_pct = None

    if request.method == "POST":
        # Read inputs
        age = float(request.form.get("age", 0))
        bmi = float(request.form.get("bmi", 0))
        exercise = int(request.form.get("exercise", 0))
        sys_bp = float(request.form.get("sys_bp", 0))
        dia_bp = float(request.form.get("dia_bp", 0))
        heart_rate = float(request.form.get("heart_rate", 0))

        form_vals = {
            "age": age,
            "bmi": bmi,
            "exercise": exercise,
            "sys_bp": sys_bp,
            "dia_bp": dia_bp,
            "heart_rate": heart_rate,
        }

        # Predict
        risk_class, proba = predict_risk(age, bmi, exercise, sys_bp, dia_bp, heart_rate)
        probability_pct = round(proba * 100, 1)
        prediction = True

        # Save to Postgres
        if engine:
            with engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO predictions
                        (created_at, age, bmi, exercise, sys_bp, dia_bp, heart_rate, risk_class, probability)
                        VALUES (:created_at, :age, :bmi, :exercise, :sys_bp, :dia_bp, :heart_rate, :risk_class, :probability)
                    """),
                    {
                        "created_at": datetime.utcnow(),
                        "age": age,
                        "bmi": bmi,
                        "exercise": exercise,
                        "sys_bp": sys_bp,
                        "dia_bp": dia_bp,
                        "heart_rate": heart_rate,
                        "risk_class": risk_class,
                        "probability": float(proba),
                    },
                )

    return render_template(
        "index.html",
        form_vals=form_vals,
        prediction=prediction,
        risk_class=risk_class,
        probability=probability_pct,
    )


@app.route("/history")
def history():
    init_db()

    rows = []
    if engine:
        with engine.begin() as conn:
            result = conn.execute(text("""
                SELECT created_at, age, bmi, exercise, sys_bp, dia_bp, heart_rate, risk_class, probability
                FROM predictions
                ORDER BY created_at DESC
                LIMIT 30;
            """))
            for r in result.fetchall():
                rows.append({
                    "created_at": r[0],
                    "age": r[1],
                    "bmi": r[2],
                    "exercise": r[3],
                    "sys_bp": r[4],
                    "dia_bp": r[5],
                    "heart_rate": r[6],
                    "risk_class": r[7],
                    "probability": round(float(r[8]) * 100, 1),
                })

    return render_template("history.html", rows=rows)


if __name__ == "__main__":
    app.run(debug=True)
