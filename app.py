import os
from datetime import datetime

import numpy as np
import joblib

from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# -------------------------
# App + DB Setup
# -------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev")

db_url = os.getenv("DATABASE_URL")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

if not db_url:
    db_url = "sqlite:///local.db"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# -------------------------
# Load Model
# -------------------------
MODEL_PATH = "demo_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

FEATURES = ["age", "bmi", "exercise_level", "sys_bp", "dia_bp", "heart_rate"]


# -------------------------
# Database Model
# -------------------------
class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.Integer, nullable=False)

    sys_bp = db.Column(db.Float, nullable=False)
    dia_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    label = db.Column(db.String(30), nullable=False)
    probability = db.Column(db.Float, nullable=False)

    source = db.Column(db.String(20), default="server", nullable=False)


# -------------------------
# Helpers
# -------------------------
def parse_exercise_level(val: str) -> int:
    if val is None:
        return 0
    s = str(val).strip().lower()
    if s in ["0", "low"]:
        return 0
    if s in ["1", "moderate", "medium"]:
        return 1
    if s in ["2", "high"]:
        return 2
    return 0


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def predict_from_inputs(age, bmi, exercise_level, sys_bp, dia_bp, heart_rate):
    if model is None:
        return "Model not loaded", 0.0

    X = np.array([[age, bmi, exercise_level, sys_bp, dia_bp, heart_rate]], dtype=float)
    pred = int(model.predict(X)[0])
    prob_high = float(model.predict_proba(X)[0][1])

    label = "High Risk" if pred == 1 else "Low Risk"
    return label, prob_high


# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            age = safe_float(request.form.get("age"))
            bmi = safe_float(request.form.get("bmi"))
            exercise_level = parse_exercise_level(request.form.get("exercise_level"))
            sys_bp = safe_float(request.form.get("sys_bp") or request.form.get("systolic_bp"))
            dia_bp = safe_float(request.form.get("dia_bp") or request.form.get("diastolic_bp"))
            heart_rate = safe_float(request.form.get("heart_rate"))

            label, prob_high = predict_from_inputs(
                age, bmi, exercise_level, sys_bp, dia_bp, heart_rate
            )

            row = Prediction(
                age=age,
                bmi=bmi,
                exercise_level=exercise_level,
                sys_bp=sys_bp,
                dia_bp=dia_bp,
                heart_rate=heart_rate,
                label=label,
                probability=prob_high,
                source="server",
            )
            db.session.add(row)
            db.session.commit()

            result = label
            probability = round(prob_high * 100, 2)

        except Exception as e:
            db.session.rollback()
            error = str(e)

    return render_template(
        "index.html", result=result, probability=probability, error=error
    )


@app.route("/history", methods=["GET"])
def history():
    rows = (
        Prediction.query.order_by(Prediction.created_at.desc())
        .limit(100)
        .all()
    )
    return render_template("history.html", rows=rows)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(force=True) or {}

        age = safe_float(payload.get("age"))
        bmi = safe_float(payload.get("bmi"))
        exercise_level = parse_exercise_level(payload.get("exercise_level"))
        sys_bp = safe_float(payload.get("sys_bp") or payload.get("systolic_bp"))
        dia_bp = safe_float(payload.get("dia_bp") or payload.get("diastolic_bp"))
        heart_rate = safe_float(payload.get("heart_rate"))

        label, prob_high = predict_from_inputs(
            age, bmi, exercise_level, sys_bp, dia_bp, heart_rate
        )

        row = Prediction(
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            sys_bp=sys_bp,
            dia_bp=dia_bp,
            heart_rate=heart_rate,
            label=label,
            probability=prob_high,
            source="api",
        )
        db.session.add(row)
        db.session.commit()

        return jsonify({"label": label, "probability": round(prob_high, 6)})

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 400


@app.get("/healthz")
def healthz():
    return {"ok": True}


if __name__ == "__main__":
    app.run(debug=True)
