import os
import uuid
import json
import logging
import datetime as dt
from functools import wraps

import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

# ==============================
# App Setup
# ==============================

app = Flask(__name__)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")

db_url = os.getenv("DATABASE_URL")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url or "sqlite:///local.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ==============================
# Logging (Structured JSON)
# ==============================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("early-risk-alert-ai")


def log_event(event_name, **kwargs):
    payload = {"event": event_name, **kwargs}
    logger.info(json.dumps(payload))


# ==============================
# Rate Limiting (Simple Memory)
# ==============================

RATE_LIMIT = 30  # requests per minute
rate_store = {}


def rate_limited(client_id):
    now = dt.datetime.utcnow()
    window_start = now - dt.timedelta(minutes=1)

    if client_id not in rate_store:
        rate_store[client_id] = []

    rate_store[client_id] = [t for t in rate_store[client_id] if t > window_start]

    if len(rate_store[client_id]) >= RATE_LIMIT:
        return True

    rate_store[client_id].append(now)
    return False


# ==============================
# API Key Protection
# ==============================

VALID_API_KEYS = os.getenv("API_KEYS", "mobile_app_key_1").split(",")


def require_api_key(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key not in VALID_API_KEYS:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)

    return wrapper


# ==============================
# DB Models
# ==============================

class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.String(64), nullable=False)
    user_id = db.Column(db.Integer, nullable=True)  # Allow anonymous
    created_at = db.Column(db.DateTime, nullable=False)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.String(20), nullable=False)
    systolic_bp = db.Column(db.Float, nullable=False)
    diastolic_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    risk_label = db.Column(db.String(50), nullable=False)
    probability = db.Column(db.Float, nullable=False)


# ==============================
# Ensure Schema Safe
# ==============================

def ensure_db_schema():
    with app.app_context():
        db.create_all()
        try:
            db.session.execute(
                text("ALTER TABLE prediction ALTER COLUMN user_id DROP NOT NULL;")
            )
            db.session.commit()
        except Exception:
            db.session.rollback()


ensure_db_schema()

# ==============================
# Load Model
# ==============================

MODEL_PATH = "demo_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# ==============================
# Health Endpoint
# ==============================

@app.route("/healthz")
def healthz():
    db_ok = True
    try:
        db.session.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    return jsonify({
        "status": "ok",
        "db_ok": db_ok,
        "model_loaded": model is not None,
        "uptime_seconds": int((dt.datetime.utcnow() - dt.datetime.utcnow()).total_seconds())
    })


# ==============================
# Prediction Endpoint
# ==============================

@app.route("/api/predict", methods=["POST"])
@require_api_key
def predict():

    data = request.get_json()
    client_id = request.headers.get("X-Client-Id")

    if not client_id:
        return jsonify({"error": "Missing Client ID"}), 400

    if rate_limited(client_id):
        return jsonify({"error": "Rate limit exceeded"}), 429

    try:
        age = float(data["age"])
        bmi = float(data["bmi"])
        exercise_level = data["exercise_level"]
        systolic_bp = float(data["systolic_bp"])
        diastolic_bp = float(data["diastolic_bp"])
        heart_rate = float(data["heart_rate"])
    except Exception:
        return jsonify({"error": "Invalid input"}), 400

    if model:
        input_data = np.array([[age, bmi, 1, systolic_bp, diastolic_bp, heart_rate]])
        pred = int(model.predict(input_data)[0])
        prob = float(model.predict_proba(input_data)[0][1])
        risk_label = "High Risk" if pred == 1 else "Low Risk"
    else:
        risk_label = "Low Risk"
        prob = 0.25

    prediction = Prediction(
        client_id=client_id,
        user_id=None,
        created_at=dt.datetime.utcnow(),
        age=age,
        bmi=bmi,
        exercise_level=exercise_level,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        heart_rate=heart_rate,
        risk_label=risk_label,
        probability=round(prob * 100, 2),
    )

    db.session.add(prediction)
    db.session.commit()

    log_event(
        "prediction_created",
        client_id=client_id,
        risk=risk_label,
        probability=prob
    )

    return jsonify({
        "risk_label": risk_label,
        "probability": round(prob * 100, 2)
    })


# ==============================
# Run
# ==============================

if __name__ == "__main__":
    app.run(debug=True)
