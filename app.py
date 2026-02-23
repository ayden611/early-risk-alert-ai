import os
from datetime import datetime

import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy

# --------------------------------------------------
# App + DB Setup
# --------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev")

db_url = os.getenv("DATABASE_URL") or os.getenv("DB_URL")

# Render can provide "postgres://", SQLAlchemy needs "postgresql://"
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

# Fallback for local dev
if not db_url:
    db_url = "sqlite:///local.db"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# --------------------------------------------------
# Model Load
# --------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "demo_model.pkl")

try:
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
except Exception:
    model = None


# --------------------------------------------------
# DB Model
# --------------------------------------------------

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


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def normalize_exercise(raw) -> tuple[str, float]:
    """
    Returns: (label_string, numeric_value)
      label_string: "Low" | "Moderate" | "High"
      numeric_value: 0.0 | 1.0 | 2.0
    Accepts: "low", "moderate", "high", "0","1","2"
    """
    if raw is None:
        return ("Low", 0.0)

    s = str(raw).strip().lower()

    # numeric inputs
    if s in {"0", "0.0"}:
        return ("Low", 0.0)
    if s in {"1", "1.0"}:
        return ("Moderate", 1.0)
    if s in {"2", "2.0"}:
        return ("High", 2.0)

    # text inputs
    if s.startswith("mod"):
        return ("Moderate", 1.0)
    if s.startswith("high"):
        return ("High", 2.0)

    # default
    return ("Low", 0.0)


def predict_risk(age: float, bmi: float, exercise_num: float, systolic_bp: float, diastolic_bp: float, heart_rate: float):
    """
    Returns: (label_string, prob_high_float_0_to_1)
    """
    if model is None:
        raise RuntimeError("Model file demo_model.pkl not found or failed to load.")

    x = np.array([[age, bmi, exercise_num, systolic_bp, diastolic_bp, heart_rate]], dtype=float)

    pred = int(model.predict(x)[0])

    # Try to get probability for "High Risk" as class 1
    prob_high = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)[0]
        # If model classes are [0,1], class 1 is High
        if len(proba) >= 2:
            prob_high = float(proba[1])
        else:
            prob_high = float(proba[0])
    else:
        # Fallback if no proba
        prob_high = 1.0 if pred == 1 else 0.0

    label = "High Risk" if pred == 1 else "Low Risk"
    return label, prob_high


def safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except Exception:
        return float(default)


# --------------------------------------------------
# Routes
# --------------------------------------------------

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
            age = safe_float(request.form.get("age"), 0)
            bmi = safe_float(request.form.get("bmi"), 0)

            ex_label, ex_num = normalize_exercise(request.form.get("exercise_level", "low"))

            systolic_bp = safe_float(request.form.get("systolic_bp"), 0)
            diastolic_bp = safe_float(request.form.get("diastolic_bp"), 0)
            heart_rate = safe_float(request.form.get("heart_rate"), 0)

            label, prob_high = predict_risk(
                age=age,
                bmi=bmi,
                exercise_num=ex_num,
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                heart_rate=heart_rate,
            )

            # Save to DB
            row = Prediction(
                age=age,
                bmi=bmi,
                exercise_level=ex_label,
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                heart_rate=heart_rate,
                risk_label=label,
                prob_high=prob_high,
            )
            db.session.add(row)
            db.session.commit()

            prediction = label
            probability = round(prob_high * 100, 2)

        except Exception as e:
            db.session.rollback()
            error = f"{type(e).__name__}: {e}"

    # Assumes you already have templates/index.html
    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error,
    )


@app.get("/history")
def history():
    rows = Prediction.query.order_by(Prediction.created_at.desc()).limit(50).all()
    # Assumes you already have templates/history.html
    return render_template("history.html", rows=rows)


# API route for Instant (JS) mode (also saves to DB)
@app.post("/api/predict")
def api_predict():
    try:
        data = request.get_json(force=True) or {}

        age = safe_float(data.get("age"), 0)
        bmi = safe_float(data.get("bmi"), 0)

        ex_label, ex_num = normalize_exercise(data.get("exercise_level", "low"))

        systolic_bp = safe_float(data.get("systolic_bp"), 0)
        diastolic_bp = safe_float(data.get("diastolic_bp"), 0)
        heart_rate = safe_float(data.get("heart_rate"), 0)

        label, prob_high = predict_risk(
            age=age,
            bmi=bmi,
            exercise_num=ex_num,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
        )

        row = Prediction(
            age=age,
            bmi=bmi,
            exercise_level=ex_label,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
            risk_label=label,
            prob_high=prob_high,
        )
        db.session.add(row)
        db.session.commit()

        return jsonify(
            ok=True,
            prediction=label,
            probability=round(prob_high * 100, 2),
        )

    except Exception as e:
        db.session.rollback()
        return jsonify(ok=False, error=f"{type(e).__name__}: {e}"), 500


# Simple debug endpoint (so /debug is NOT 404 anymore)
@app.get("/debug")
def debug():
    try:
        count = Prediction.query.count()
        latest = Prediction.query.order_by(Prediction.created_at.desc()).first()
        latest_dict = None
        if latest:
            latest_dict = {
                "created_at": latest.created_at.isoformat(),
                "age": latest.age,
                "bmi": latest.bmi,
                "exercise_level": latest.exercise_level,
                "systolic_bp": latest.systolic_bp,
                "diastolic_bp": latest.diastolic_bp,
                "heart_rate": latest.heart_rate,
                "risk_label": latest.risk_label,
                "prob_high": latest.prob_high,
            }

        return jsonify(
            ok=True,
            database_uri=app.config["SQLALCHEMY_DATABASE_URI"],
            model_loaded=bool(model is not None),
            prediction_count=count,
            latest=latest_dict,
        )
    except Exception as e:
        return jsonify(ok=False, error=f"{type(e).__name__}: {e}"), 500


# --------------------------------------------------
# Local run (Render uses gunicorn, so this won't run there)
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=True)
