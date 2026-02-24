import os
import json
from datetime import datetime

import numpy as np
import joblib

from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy

# --------------------------------------------------
# App Setup
# --------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev")

db_url = os.getenv("DATABASE_URL")

# Render sometimes uses postgres:// (old). SQLAlchemy wants postgresql://
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

# Local fallback
if not db_url:
    db_url = "sqlite:///local.db"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# --------------------------------------------------
# DB Model
# --------------------------------------------------

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.Integer, nullable=False)  # 0/1/2
    sys_bp = db.Column(db.Float, nullable=False)
    dia_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    pred_label = db.Column(db.String(20), nullable=False)
    prob_high = db.Column(db.Float, nullable=False)


with app.app_context():
    db.create_all()

# --------------------------------------------------
# Load Model
# --------------------------------------------------

MODEL_PATH = "demo_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

FEATURES = ["age", "bmi", "exercise_level", "sys_bp", "dia_bp", "heart_rate"]

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _to_float(value, field_name):
    try:
        return float(value)
    except Exception:
        raise ValueError(f"Invalid {field_name}. Must be a number.")

def _to_int(value, field_name):
    try:
        return int(float(value))
    except Exception:
        raise ValueError(f"Invalid {field_name}. Must be 0, 1, or 2.")

def parse_exercise_level(value):
    """
    Accepts: 0/1/2 OR strings: low/moderate/high
    """
    if value is None:
        raise ValueError("exercise_level is required.")

    # numeric path
    if str(value).strip().replace(".", "", 1).isdigit():
        lvl = _to_int(value, "exercise_level")
        if lvl not in (0, 1, 2):
            raise ValueError("exercise_level must be 0, 1, or 2.")
        return lvl

    s = str(value).strip().lower()
    mapping = {"low": 0, "moderate": 1, "medium": 1, "high": 2}
    if s in mapping:
        return mapping[s]

    raise ValueError("exercise_level must be Low/Moderate/High (or 0/1/2).")

def safe_get_json(req):
    """
    Makes Instant JS reliable even if headers are wrong.
    Tries:
      - request.get_json(silent=True)
      - json.loads(request.data)
      - form fallback
    """
    data = req.get_json(silent=True)
    if isinstance(data, dict):
        return data

    raw = (req.data or b"").decode("utf-8", errors="ignore").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # final fallback: treat as form data
    if req.form:
        return dict(req.form)

    return {}

def build_features(payload: dict):
    """
    Payload keys accepted:
      age, bmi, exercise_level, sys_bp, dia_bp, heart_rate
    """
    age = _to_float(payload.get("age"), "age")
    bmi = _to_float(payload.get("bmi"), "bmi")
    sys_bp = _to_float(payload.get("sys_bp"), "sys_bp")
    dia_bp = _to_float(payload.get("dia_bp"), "dia_bp")
    heart_rate = _to_float(payload.get("heart_rate"), "heart_rate")
    exercise_level = parse_exercise_level(payload.get("exercise_level"))

    X = np.array([[age, bmi, exercise_level, sys_bp, dia_bp, heart_rate]], dtype=float)
    return age, bmi, exercise_level, sys_bp, dia_bp, heart_rate, X

def run_model(X):
    if model is None:
        raise RuntimeError("Model file demo_model.pkl not found on server.")

    pred = int(model.predict(X)[0])

    # Prefer predict_proba if available
    prob_high = None
    if hasattr(model, "predict_proba"):
        prob_high = float(model.predict_proba(X)[0][1])
    else:
        # fallback if model doesn't support proba
        prob_high = 1.0 if pred == 1 else 0.0

    label = "High Risk" if pred == 1 else "Low Risk"
    return label, prob_high

def save_prediction(age, bmi, exercise_level, sys_bp, dia_bp, heart_rate, label, prob_high):
    try:
        row = Prediction(
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            sys_bp=sys_bp,
            dia_bp=dia_bp,
            heart_rate=heart_rate,
            pred_label=label,
            prob_high=prob_high,
        )
        db.session.add(row)
        db.session.commit()
    except Exception:
        # Don't crash the app if DB write fails
        db.session.rollback()

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    # page renders both forms
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_server():
    """
    Server Submit (Reload) form POST.
    Make sure your HTML form action="/predict" method="POST"
    """
    try:
        payload = dict(request.form)
        age, bmi, exercise_level, sys_bp, dia_bp, heart_rate, X = build_features(payload)
        label, prob_high = run_model(X)

        save_prediction(age, bmi, exercise_level, sys_bp, dia_bp, heart_rate, label, prob_high)

        return render_template(
            "index.html",
            prediction=label,
            probability=round(prob_high * 100, 2),
        )
    except Exception as e:
        # Render error cleanly on the page instead of default 400
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}",
            probability=0,
        ), 200

@app.route("/api/predict", methods=["POST"])
def predict_api():
    """
    Instant JS calls this route.
    It accepts JSON OR form data (for reliability).
    """
    try:
        payload = safe_get_json(request)
        age, bmi, exercise_level, sys_bp, dia_bp, heart_rate, X = build_features(payload)
        label, prob_high = run_model(X)

        save_prediction(age, bmi, exercise_level, sys_bp, dia_bp, heart_rate, label, prob_high)

        return jsonify({
            "ok": True,
            "prediction": label,
            "probability": round(prob_high * 100, 2),
        }), 200

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "prediction": f"Error: {str(e)}",
            "probability": 0
        }), 200  # keep 200 so your UI can always display message

@app.route("/history", methods=["GET"])
def history():
    rows = Prediction.query.order_by(Prediction.created_at.desc()).limit(100).all()
    return render_template("history.html", rows=rows)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "model_loaded": model is not None}), 200


if __name__ == "__main__":
    # local dev only
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
