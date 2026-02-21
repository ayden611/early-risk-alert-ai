import os
import json
from datetime import datetime

import joblib
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for,
    flash, session, make_response
)
from sqlalchemy import create_engine, text

# -----------------------------
# App setup
# -----------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

# Simple auth (set these in Render ENV)
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "changeme")

# Model + metrics files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "demo_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

model = joblib.load(MODEL_PATH)

model_metrics = {}
if os.path.exists(METRICS_PATH):
    try:
        with open(METRICS_PATH, "r") as f:
            model_metrics = json.load(f)
    except Exception:
        model_metrics = {}

# -----------------------------
# Database (Postgres on Render)
# -----------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///local.db"

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP NOT NULL,
                age FLOAT,
                bmi FLOAT,
                exercise_level TEXT,
                systolic_bp FLOAT,
                diastolic_bp FLOAT,
                heart_rate FLOAT,
                risk_label TEXT,
                probability FLOAT
            )
        """))

init_db()

# -----------------------------
# Helpers
# -----------------------------
def is_logged_in() -> bool:
    return bool(session.get("logged_in"))

def require_login():
    if not is_logged_in():
        return redirect(url_for("login"))
    return None

def _to_float(name: str) -> float:
    return float(request.form[name])

def _exercise_to_num(exercise_str: str) -> float:
    mapping = {"Low": 0.0, "Moderate": 1.0, "High": 2.0}
    return mapping.get(exercise_str, 0.0)

def _predict(age, bmi, exercise_level_str, systolic_bp, diastolic_bp, heart_rate):
    ex_num = _exercise_to_num(exercise_level_str)
    X = np.array([[age, bmi, ex_num, systolic_bp, diastolic_bp, heart_rate]], dtype=float)
    pred = int(model.predict(X)[0])
    proba_high = float(model.predict_proba(X)[0][1])
    label = "High" if pred == 1 else "Low"
    return label, proba_high

# -----------------------------
# Routes
# -----------------------------
@app.get("/login")
def login():
    return render_template("login.html")

@app.post("/login")
def login_post():
    username = request.form.get("username", "")
    password = request.form.get("password", "")

    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session["logged_in"] = True
        return redirect(url_for("home"))

    flash("Invalid login.")
    return redirect(url_for("login"))

@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/", methods=["GET", "POST"])
def home():
    gate = require_login()
    if gate:
        return gate

    prediction = None
    probability_pct = None

    form_defaults = {
        "age": "",
        "bmi": "",
        "exercise_level": "Low",
        "systolic_bp": "",
        "diastolic_bp": "",
        "heart_rate": "",
        "ack": False
    }

    if request.method == "POST":
        ack = request.form.get("ack") == "on"
        if not ack:
            flash("Please check the acknowledgement box before predicting.")
            return render_template("index.html", prediction=None, probability_pct=None, form=form_defaults)

        try:
            age = _to_float("age")
            bmi = _to_float("bmi")
            exercise_level = request.form.get("exercise_level", "Low")
            systolic_bp = _to_float("systolic_bp")
            diastolic_bp = _to_float("diastolic_bp")
            heart_rate = _to_float("heart_rate")

            label, proba_high = _predict(age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate)

            prediction = label
            probability_pct = round(proba_high * 100.0, 1)

            with engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO predictions (
                            created_at, age, bmi, exercise_level,
                            systolic_bp, diastolic_bp, heart_rate,
                            risk_label, probability
                        ) VALUES (
                            :created_at, :age, :bmi, :exercise_level,
                            :systolic_bp, :diastolic_bp, :heart_rate,
                            :risk_label, :probability
                        )
                    """),
                    {
                        "created_at": datetime.utcnow(),
                        "age": age,
                        "bmi": bmi,
                        "exercise_level": exercise_level,
                        "systolic_bp": systolic_bp,
                        "diastolic_bp": diastolic_bp,
                        "heart_rate": heart_rate,
                        "risk_label": label,
                        "probability": float(proba_high),
                    }
                )

            form_defaults.update({
                "age": age,
                "bmi": bmi,
                "exercise_level": exercise_level,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "heart_rate": heart_rate,
                "ack": True
            })

        except Exception as e:
            flash(f"Input error: {e}")

    return render_template(
        "index.html",
        prediction=prediction,
        probability_pct=probability_pct,
        form=form_defaults
    )

@app.get("/history")
def history():
    gate = require_login()
    if gate:
        return gate

    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT id, created_at, age, bmi, exercise_level,
                       systolic_bp, diastolic_bp, heart_rate,
                       risk_label, probability
                FROM predictions
                ORDER BY created_at DESC
                LIMIT 200
            """)
        ).fetchall()

    return render_template("history.html", rows=rows)

@app.post("/history/clear")
def clear_history():
    gate = require_login()
    if gate:
        return gate

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM predictions"))
    flash("History cleared.")
    return redirect(url_for("history"))

@app.post("/history/delete/<int:row_id>")
def delete_row(row_id: int):
    gate = require_login()
    if gate:
        return gate

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM predictions WHERE id = :id"), {"id": row_id})
    flash("Deleted one record.")
    return redirect(url_for("history"))

@app.get("/download/history.csv")
def download_history_csv():
    gate = require_login()
    if gate:
        return gate

    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT created_at, age, bmi, exercise_level,
                       systolic_bp, diastolic_bp, heart_rate,
                       risk_label, probability
                FROM predictions
                ORDER BY created_at DESC
            """)
        ).fetchall()

    header = "created_at,age,bmi,exercise_level,systolic_bp,diastolic_bp,heart_rate,risk_label,probability\n"
    lines = [header]
    for r in rows:
        created_at = r[0].isoformat() if r[0] else ""
        line = f"{created_at},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]},{r[6]},{r[7]},{r[8]}\n"
        lines.append(line)

    csv_data = "".join(lines)
    resp = make_response(csv_data)
    resp.headers["Content-Type"] = "text/csv"
    resp.headers["Content-Disposition"] = "attachment; filename=prediction_history.csv"
    return resp

@app.get("/metrics")
def metrics():
    gate = require_login()
    if gate:
        return gate
    return render_template("metrics.html", metrics=model_metrics)
