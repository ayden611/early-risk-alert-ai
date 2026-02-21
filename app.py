import os
import math
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
from sqlalchemy import create_engine, text
import joblib


# ----------------------------
# APP SETUP
# ----------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")  # needed for flash messages

# ----------------------------
# DATABASE SETUP
# ----------------------------
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

engine = None
if DATABASE_URL:
    # Render Postgres often provides postgres:// which SQLAlchemy wants as postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
else:
    # Fallback for local/dev if you run without Postgres
    engine = create_engine("sqlite:///local.db", connect_args={"check_same_thread": False})


def init_db():
    """Create table if it doesn't exist."""
    if not engine:
        return

    # Works on Postgres. For sqlite, SERIAL isn't valid, but sqlite will ignore type-ish.
    # If you want perfect sqlite support, we can add a sqlite-specific DDL later.
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


# ----------------------------
# LOAD MODEL
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "demo_model.pkl")
model = None

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    # Don't crash boot; show message in UI instead
    model = None
    print("⚠️ Model load error:", repr(e))


# ----------------------------
# HELPERS
# ----------------------------
def to_float(value, field_name):
    """Convert form input to float safely."""
    if value is None:
        raise ValueError(f"Missing {field_name}")
    s = str(value).strip()
    if s == "":
        raise ValueError(f"Missing {field_name}")
    return float(s)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def predict_probability(features):
    """
    Returns probability of HIGH RISK as float in [0, 1].
    Works with models that have predict_proba OR decision_function.
    """
    if model is None:
        raise RuntimeError("Model is not loaded (demo_model.pkl missing or failed to load).")

    X = [features]  # shape (1, n_features)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        # assume binary, class 1 = high risk probability
        if len(proba) >= 2:
            return float(proba[1])
        return float(proba[0])

    if hasattr(model, "decision_function"):
        score = float(model.decision_function(X)[0])
        return sigmoid(score)

    # last resort
    pred = int(model.predict(X)[0])
    return 1.0 if pred == 1 else 0.0


def compute_result(prob_high):
    """
    Builds the dict used by index.html to show results.
    - probability shown as percent
    - confidence shown as percent (how strong the model leans)
    """
    prob_high = clamp(prob_high, 0.0, 1.0)

    risk_class = "high" if prob_high >= 0.5 else "low"
    risk_label = "High Risk" if risk_class == "high" else "Low Risk"

    # Display formatting:
    # If the model gives tiny probabilities, you were seeing 0.0%.
    # We keep the *stored* raw probability, but we clamp DISPLAY to avoid "0.0%" and "100.0%" unless truly extreme.
    display_prob = clamp(prob_high, 0.001, 0.999)  # 0.1% to 99.9% display floor/ceiling
    probability_pct = round(display_prob * 100.0, 1)

    # confidence = max(prob, 1-prob)
    conf = max(prob_high, 1.0 - prob_high)
    conf_display = clamp(conf, 0.501, 0.999)  # keep it meaningful visually
    confidence_pct = round(conf_display * 100.0, 1)

    return {
        "risk_label": risk_label,
        "risk_class": risk_class,
        "probability": probability_pct,
        "confidence": confidence_pct,
        # raw for debug / future charts:
        "raw_probability": prob_high,
    }


# ----------------------------
# ROUTES
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    # Keep form values so page re-renders with what user typed
    form = {
        "age": "",
        "bmi": "",
        "exercise_level": "",
        "systolic_bp": "",
        "diastolic_bp": "",
        "heart_rate": "",
        "ack": False,
    }

    if request.method == "POST":
        try:
            form["age"] = request.form.get("age", "")
            form["bmi"] = request.form.get("bmi", "")
            form["exercise_level"] = request.form.get("exercise_level", "")
            form["systolic_bp"] = request.form.get("systolic_bp", "")
            form["diastolic_bp"] = request.form.get("diastolic_bp", "")
            form["heart_rate"] = request.form.get("heart_rate", "")
            form["ack"] = bool(request.form.get("ack"))

            if not form["ack"]:
                raise ValueError("Please check the acknowledgement box.")

            age = to_float(form["age"], "age")
            bmi = to_float(form["bmi"], "bmi")
            systolic_bp = to_float(form["systolic_bp"], "systolic_bp")
            diastolic_bp = to_float(form["diastolic_bp"], "diastolic_bp")
            heart_rate = to_float(form["heart_rate"], "heart_rate")
            exercise_level = (form["exercise_level"] or "").strip()

            # Basic sanity clamps (prevents crazy inputs breaking stuff)
            age = clamp(age, 0, 120)
            bmi = clamp(bmi, 10, 80)
            systolic_bp = clamp(systolic_bp, 60, 260)
            diastolic_bp = clamp(diastolic_bp, 30, 160)
            heart_rate = clamp(heart_rate, 30, 220)

            # Feature order MUST match training
            features = [age, bmi, systolic_bp, diastolic_bp, heart_rate]

            prob_high = predict_probability(features)
            result = compute_result(prob_high)

            # Save prediction
            if engine:
                with engine.begin() as conn:
                    conn.execute(
                        text(
                            """
                            INSERT INTO predictions
                            (age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate,
                             risk_label, risk_class, probability, confidence)
                            VALUES
                            (:age, :bmi, :exercise_level, :systolic_bp, :diastolic_bp, :heart_rate,
                             :risk_label, :risk_class, :probability, :confidence)
                            """
                        ),
                        {
                            "age": age,
                            "bmi": bmi,
                            "exercise_level": exercise_level,
                            "systolic_bp": systolic_bp,
                            "diastolic_bp": diastolic_bp,
                            "heart_rate": heart_rate,
                            "risk_label": result["risk_label"],
                            "risk_class": result["risk_class"],
                            # store RAW probability and confidence (0..1), not the display %
                            "probability": float(result["raw_probability"]),
                            "confidence": float(max(result["raw_probability"], 1.0 - result["raw_probability"])),
                        },
                    )

        except Exception as e:
            error = str(e)
            # Friendly fallback message if it's a generic crash
            if error.lower().startswith(("runtimeerror", "valueerror")) is False and len(error) > 180:
                error = "Oops! Invalid input. Please check your values."
            # keep result None

    return render_template("index.html", result=result, error=error, form=form)


@app.route("/history", methods=["GET"])
def history():
    """
    History with filtering via query params:
      /history?risk_class=low|high
      /history?min_prob=10&max_prob=60   (probability in percent)
      /history?date_from=YYYY-MM-DD&date_to=YYYY-MM-DD
    """
    if not engine:
        return render_template("history.html", history=[], filters={})

    risk_class = (request.args.get("risk_class", "") or "").strip().lower()
    min_prob = (request.args.get("min_prob", "") or "").strip()
    max_prob = (request.args.get("max_prob", "") or "").strip()
    date_from = (request.args.get("date_from", "") or "").strip()
    date_to = (request.args.get("date_to", "") or "").strip()

    where = []
    params = {}

    if risk_class in ("low", "high"):
        where.append("risk_class = :risk_class")
        params["risk_class"] = risk_class

    # probability stored as 0..1 float
    if min_prob != "":
        try:
            where.append("probability >= :min_p")
            params["min_p"] = float(min_prob) / 100.0
        except:
            pass

    if max_prob != "":
        try:
            where.append("probability <= :max_p")
            params["max_p"] = float(max_prob) / 100.0
        except:
            pass

    # created_at is timestamptz
    if date_from:
        where.append("created_at >= :date_from")
        params["date_from"] = date_from

    if date_to:
        # include entire date_to day by using next day boundary is nicer, but keep simple:
        where.append("created_at <= :date_to")
        params["date_to"] = date_to

    where_sql = ""
    if where:
        where_sql = "WHERE " + " AND ".join(where)

    query = f"""
        SELECT *
        FROM predictions
        {where_sql}
        ORDER BY created_at DESC
        LIMIT 200
    """

    with engine.begin() as conn:
        rows = conn.execute(text(query), params).mappings().all()

    # Convert stored probability/confidence to percent for display
    history_rows = []
    for r in rows:
        rr = dict(r)
        rr["probability_pct"] = round(float(rr.get("probability", 0.0)) * 100.0, 1)
        rr["confidence_pct"] = round(float(rr.get("confidence", 0.0)) * 100.0, 1)
        history_rows.append(rr)

    filters = {
        "risk_class": risk_class,
        "min_prob": min_prob,
        "max_prob": max_prob,
        "date_from": date_from,
        "date_to": date_to,
    }

    return render_template("history.html", history=history_rows, filters=filters)


@app.route("/history/clear", methods=["POST"])
def clear_history():
    if not engine:
        flash("No database connected.", "error")
        return redirect(url_for("history"))

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM predictions"))

    flash("History cleared.", "success")
    return redirect(url_for("history"))


@app.get("/health")
def health():
    return {"ok": True, "model_loaded": bool(model), "db_connected": bool(engine)}


# IMPORTANT FOR RENDER:
# Do NOT call app.run() on Render/Gunicorn.
# Keep app as a module-level variable named "app".
