import os
from functools import wraps
from datetime import datetime

import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash

from sqlalchemy import create_engine, text


# ---------------------------
# Config
# ---------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "demo_model.pkl")

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")  # set in Render env vars

# Render Postgres sometimes provides postgres:// (SQLAlchemy wants postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Fallback (local dev) if DATABASE_URL not set
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///predictions.db"

engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)


# ---------------------------
# App
# ---------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = SECRET_KEY

model = joblib.load(MODEL_PATH)


# ---------------------------
# DB setup
# ---------------------------
def init_db():
    # Works for Postgres + SQLite
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
                created_at TIMESTAMP NOT NULL,
                age REAL NOT NULL,
                bmi REAL NOT NULL,
                exercise_level TEXT NOT NULL,
                systolic_bp REAL NOT NULL,
                diastolic_bp REAL NOT NULL,
                heart_rate REAL NOT NULL,
                risk_label TEXT NOT NULL,
                probability REAL NOT NULL
            )
        """))


# SQLite doesn't support GENERATED ALWAYS AS IDENTITY exactly.
# If SQLite is used, the above may fail. We retry with SQLite-friendly SQL.
def init_db_sqlite_safe():
    if DATABASE_URL.startswith("sqlite:///"):
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    age REAL NOT NULL,
                    bmi REAL NOT NULL,
                    exercise_level TEXT NOT NULL,
                    systolic_bp REAL NOT NULL,
                    diastolic_bp REAL NOT NULL,
                    heart_rate REAL NOT NULL,
                    risk_label TEXT NOT NULL,
                    probability REAL NOT NULL
                )
            """))


try:
    init_db()
except Exception:
    init_db_sqlite_safe()


# ---------------------------
# Auth helpers
# ---------------------------
def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper


# ---------------------------
# Routes
# ---------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "").strip()

        if u == ADMIN_USERNAME and p == ADMIN_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("home"))

        flash("Invalid login.")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    prediction = None
    probability_pct = None
    error = None

    if request.method == "POST":
        try:
            age = float(request.form["age"])
            bmi = float(request.form["bmi"])
            exercise_level = request.form["exercise_level"].strip()  # Low/Moderate/High
            systolic_bp = float(request.form["systolic_bp"])
            diastolic_bp = float(request.form["diastolic_bp"])
            heart_rate = float(request.form["heart_rate"])

            # Map exercise text -> number (keep consistent with training)
            ex_map = {"Low": 0, "Moderate": 1, "High": 2}
            ex_val = ex_map.get(exercise_level, 0)

            X = np.array([[age, bmi, ex_val, systolic_bp, diastolic_bp, heart_rate]], dtype=float)

            pred = int(model.predict(X)[0])
            prob_high = float(model.predict_proba(X)[0][1])  # class 1 = High risk

            prediction = "High" if pred == 1 else "Low"
            probability_pct = round(prob_high * 100, 1)

            # Save to DB
            with engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO predictions
                        (created_at, age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate, risk_label, probability)
                        VALUES (:created_at, :age, :bmi, :exercise_level, :systolic_bp, :diastolic_bp, :heart_rate, :risk_label, :probability)
                    """),
                    dict(
                        created_at=datetime.utcnow(),
                        age=age,
                        bmi=bmi,
                        exercise_level=exercise_level,
                        systolic_bp=systolic_bp,
                        diastolic_bp=diastolic_bp,
                        heart_rate=heart_rate,
                        risk_label=prediction,
                        probability=prob_high,  # store 0..1
                    )
                )

        except Exception as e:
            error = f"Input error: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability_pct,
        error=error
    )


@app.route("/history")
@login_required
def history():
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT created_at, age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate, risk_label, probability
                FROM predictions
                ORDER BY created_at DESC
                LIMIT 200
            """)
        ).mappings().all()

    # Convert prob to percent for display
    history_rows = []
    for r in rows:
        history_rows.append({
            "created_at": str(r["created_at"]),
            "age": float(r["age"]),
            "bmi": float(r["bmi"]),
            "exercise_level": r["exercise_level"],
            "systolic_bp": float(r["systolic_bp"]),
            "diastolic_bp": float(r["diastolic_bp"]),
            "heart_rate": float(r["heart_rate"]),
            "risk_label": r["risk_label"],
            "probability_pct": round(float(r["probability"]) * 100, 1),
        })

    return render_template("history.html", rows=history_rows)
