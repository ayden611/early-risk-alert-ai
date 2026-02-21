import os
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
from sqlalchemy import create_engine, text
import joblib

# =========================
# APP SETUP
# =========================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

# =========================
# MODEL SETUP
# =========================
MODEL_PATH = os.environ.get("MODEL_PATH", "demo_model.pkl")

model = None
model_load_error = None
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    model_load_error = str(e)

# =========================
# DATABASE SETUP
# =========================
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

engine = None
db_error = None

try:
    if DATABASE_URL:
        # Render Postgres sometimes uses postgres:// but SQLAlchemy wants postgresql://
        if DATABASE_URL.startswith("postgres://"):
            DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    else:
        # local/dev fallback
        engine = create_engine(
            "sqlite:///local.db",
            connect_args={"check_same_thread": False}
        )
except Exception as e:
    engine = None
    db_error = str(e)


def init_db():
    """Create table if it doesn't exist."""
    if not engine:
        return
    ddl = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        age REAL,
        bmi REAL,
        exercise_level TEXT,
        systolic_bp REAL,
        diastolic_bp REAL,
        heart_rate REAL,
        risk_label TEXT,
        probability REAL
    )
    """
    # Postgres doesn't support AUTOINCREMENT; but this DDL is fine for SQLite.
    # For Postgres, we’ll use a compatible table creation:
    if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
        ddl = """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMP NOT NULL,
            age DOUBLE PRECISION,
            bmi DOUBLE PRECISION,
            exercise_level TEXT,
            systolic_bp DOUBLE PRECISION,
            diastolic_bp DOUBLE PRECISION,
            heart_rate DOUBLE PRECISION,
            risk_label TEXT,
            probability DOUBLE PRECISION
        )
        """
    with engine.begin() as conn:
        conn.execute(text(ddl))


# Initialize DB on startup
try:
    init_db()
except Exception as e:
    db_error = str(e)


# =========================
# HELPERS
# =========================
def to_float(value, default=None):
    try:
        if value is None:
            return default
        v = str(value).strip()
        if v == "":
            return default
        return float(v)
    except Exception:
        return default


def clamp(x, lo, hi):
    if x is None:
        return None
    return max(lo, min(hi, x))


def predict_probability(age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate):
    """
    Uses the trained sklearn pipeline (StandardScaler + LogisticRegression)
    saved in demo_model.pkl.
    """
    if model is None:
        raise RuntimeError(f"Model not loaded: {model_load_error}")

    # exercise_level is stored as Low/Moderate/High in UI.
    # Our training file may have mapped it to numeric at training time,
    # BUT the final model pipeline expects numeric features.
    # We'll map here to be safe.
    ex_map = {"low": 0, "moderate": 1, "high": 2}
    ex_val = ex_map.get(str(exercise_level).strip().lower(), 0)

    X = [[age, bmi, ex_val, systolic_bp, diastolic_bp, heart_rate]]
    proba = model.predict_proba(X)[0][1]  # probability of class 1 (high risk)
    return float(proba)


def risk_label_from_proba(p):
    # Simple threshold; you can tune later
    return "High" if p >= 0.5 else "Low"


# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Checkbox acknowledgment
        agree = request.form.get("agree")
        if not agree:
            flash("Please confirm the awareness-only disclaimer checkbox.")
            return redirect(url_for("home"))

        age = clamp(to_float(request.form.get("age"), None), 0, 120)
        bmi = clamp(to_float(request.form.get("bmi"), None), 5, 80)
        exercise_level = request.form.get("exercise_level", "Low")

        systolic_bp = clamp(to_float(request.form.get("systolic_bp"), None), 50, 300)
        diastolic_bp = clamp(to_float(request.form.get("diastolic_bp"), None), 30, 200)
        heart_rate = clamp(to_float(request.form.get("heart_rate"), None), 30, 220)

        # Basic validation
        missing = [x is None for x in [age, bmi, systolic_bp, diastolic_bp, heart_rate]]
        if any(missing):
            flash("Please fill out all fields with valid numbers.")
            return redirect(url_for("home"))

        try:
            proba = predict_probability(age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate)
            label = risk_label_from_proba(proba)
        except Exception as e:
            flash(f"Prediction error: {e}")
            return redirect(url_for("home"))

        # Save to DB (best effort)
        try:
            if engine:
                with engine.begin() as conn:
                    if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
                        conn.execute(
                            text("""
                                INSERT INTO predictions
                                (created_at, age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate, risk_label, probability)
                                VALUES (:created_at, :age, :bmi, :exercise_level, :systolic_bp, :diastolic_bp, :heart_rate, :risk_label, :probability)
                            """),
                            {
                                "created_at": datetime.utcnow(),
                                "age": age,
                                "bmi": bmi,
                                "exercise_level": str(exercise_level),
                                "systolic_bp": systolic_bp,
                                "diastolic_bp": diastolic_bp,
                                "heart_rate": heart_rate,
                                "risk_label": label,
                                "probability": proba,
                            }
                        )
                    else:
                        conn.execute(
                            text("""
                                INSERT INTO predictions
                                (created_at, age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate, risk_label, probability)
                                VALUES (:created_at, :age, :bmi, :exercise_level, :systolic_bp, :diastolic_bp, :heart_rate, :risk_label, :probability)
                            """),
                            {
                                "created_at": datetime.utcnow().isoformat(),
                                "age": age,
                                "bmi": bmi,
                                "exercise_level": str(exercise_level),
                                "systolic_bp": systolic_bp,
                                "diastolic_bp": diastolic_bp,
                                "heart_rate": heart_rate,
                                "risk_label": label,
                                "probability": proba,
                            }
                        )
        except Exception as e:
            # Don’t kill the app if DB fails; just show message
            flash(f"Note: Could not save history (DB issue): {e}")

        # Render result on the home page
        return render_template(
            "index.html",
            result={
                "label": label,
                "probability": round(proba * 100, 1),
            }
        )

    return render_template("index.html")


@app.route("/history")
def history():
    """
    Robust history route:
    - If DB is missing/broken, show empty list and an error message.
    - Always send a list of dicts to the template.
    """
    rows = []
    error = None

    try:
        if not engine:
            raise RuntimeError(db_error or "Database engine not available.")

        with engine.connect() as conn:
            # Works for both sqlite and postgres
            result = conn.execute(text("""
                SELECT created_at, age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate, risk_label, probability
                FROM predictions
                ORDER BY created_at DESC
                LIMIT 50
            """))

            for r in result.mappings():
                rows.append({
                    "created_at": str(r.get("created_at")),
                    "age": r.get("age"),
                    "bmi": r.get("bmi"),
                    "exercise_level": r.get("exercise_level"),
                    "systolic_bp": r.get("systolic_bp"),
                    "diastolic_bp": r.get("diastolic_bp"),
                    "heart_rate": r.get("heart_rate"),
                    "risk_label": r.get("risk_label"),
                    "probability": r.get("probability"),
                })

    except Exception as e:
        error = str(e)

    return render_template("history.html", rows=rows, error=error)


@app.route("/clear-history", methods=["POST"])
def clear_history():
    try:
        if not engine:
            raise RuntimeError(db_error or "Database engine not available.")
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM predictions"))
        flash("History cleared.")
    except Exception as e:
        flash(f"Could not clear history: {e}")
    return redirect(url_for("history"))


@app.route("/health")
def health():
    """
    Health check endpoint so Render / you can confirm status quickly.
    """
    return {
        "ok": True,
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_error": model_load_error,
        "db_engine": bool(engine),
        "db_error": db_error,
        "has_database_url": bool(os.environ.get("DATABASE_URL")),
    }, 200


# =========================
# LOCAL RUN (optional)
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
