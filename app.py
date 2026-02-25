import os
import json
import uuid
import logging
from datetime import datetime, timezone

import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text


# ----------------------------
# App setup
# ----------------------------
app = Flask(__name__)

# Secrets / config
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL or "sqlite:///local.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("early-risk-alert-ai")

# Model paths
MODEL_PATH = os.getenv("MODEL_PATH", "demo_model.pkl")
MODEL_META_PATH = os.getenv("MODEL_META_PATH", "model_meta.json")

# API key config
API_KEYS_ENV = os.getenv("API_KEYS", "")
API_KEYS = {k.strip() for k in API_KEYS_ENV.replace("\n", ",").split(",") if k.strip()}
REQUIRE_API_KEY_FOR_API = os.getenv("REQUIRE_API_KEY_FOR_API", "true").lower() in ("1", "true", "yes")


# ----------------------------
# DB model
# ----------------------------
class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    # request identity
    client_id = db.Column(db.String(128), nullable=True)
    user_id = db.Column(db.Integer, nullable=True)

    # inputs
    age = db.Column(db.Float, nullable=True)
    bmi = db.Column(db.Float, nullable=True)

    # IMPORTANT:
    # store numeric so Postgres "double precision" columns won't choke
    exercise_level = db.Column(db.Float, nullable=True)

    systolic_bp = db.Column(db.Float, nullable=True)
    diastolic_bp = db.Column(db.Float, nullable=True)
    heart_rate = db.Column(db.Float, nullable=True)

    # outputs
    risk_label = db.Column(db.String(32), nullable=False, default="Unknown")
    probability = db.Column(db.Float, nullable=False, default=0.0)


# ----------------------------
# Utilities
# ----------------------------
def load_model():
    model = None
    model_error = None
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            model_error = f"Failed to load model from {MODEL_PATH}: {e}"
            logger.exception(model_error)
    else:
        model_error = f"Model file not found at {MODEL_PATH}"
    return model, model_error


MODEL, MODEL_ERROR = load_model()


def parse_exercise_level(val):
    """
    Accepts:
      - "Low" / "Moderate" / "High"
      - 0/1/2
      - numeric strings
    Returns float (safe for DB + model)
    """
    if val is None:
        return None

    # already numeric
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip().lower()
    mapping = {
        "low": 0.0,
        "l": 0.0,
        "0": 0.0,
        "moderate": 1.0,
        "mod": 1.0,
        "m": 1.0,
        "1": 1.0,
        "high": 2.0,
        "h": 2.0,
        "2": 2.0,
    }
    if s in mapping:
        return mapping[s]

    # try numeric conversion (e.g. "1.5")
    try:
        return float(s)
    except Exception:
        return None


def require_api_key_if_needed():
    if not REQUIRE_API_KEY_FOR_API:
        return None  # allow without key

    key = request.headers.get("X-API-Key", "").strip()
    if not key or key not in API_KEYS:
        return jsonify({"error": "unauthorized", "message": "Missing or invalid API key"}), 401
    return None


def ensure_schema():
    """
    Creates tables. If you already have the table, this is safe.
    """
    db.create_all()

    # If old DB exists with missing columns, try to add them.
    # (Safe IF NOT EXISTS)
    try:
        with db.engine.begin() as conn:
            conn.execute(text("ALTER TABLE prediction ADD COLUMN IF NOT EXISTS client_id VARCHAR(128);"))
            conn.execute(text("ALTER TABLE prediction ADD COLUMN IF NOT EXISTS user_id INTEGER;"))
            conn.execute(text("ALTER TABLE prediction ADD COLUMN IF NOT EXISTS probability DOUBLE PRECISION DEFAULT 0;"))
            conn.execute(text("ALTER TABLE prediction ADD COLUMN IF NOT EXISTS risk_label VARCHAR(32) DEFAULT 'Unknown';"))
    except Exception as e:
        logger.warning("Schema alter skipped/failed (may be SQLite): %s", e)


with app.app_context():
    ensure_schema()


# ----------------------------
# Routes
# ----------------------------
@app.get("/healthz")
def healthz():
    # basic db check
    db_ok = True
    db_error = None
    try:
        with db.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        db_ok = False
        db_error = str(e)

    return jsonify(
        {
            "status": "ok",
            "model_loaded": MODEL is not None,
            "model_error": MODEL_ERROR,
            "db_ok": db_ok,
            "db_error": db_error,
            "uptime_seconds": 0,
        }
    )


@app.post("/api/predict")
def api_predict():
    # API key gate (if enabled)
    auth_resp = require_api_key_if_needed()
    if auth_resp:
        return auth_resp

    payload = request.get_json(silent=True) or {}

    # Client id: allow header or payload, else generate
    client_id = (request.headers.get("X-Client-Id") or payload.get("client_id") or str(uuid.uuid4())).strip()

    # Read fields
    try:
        age = float(payload.get("age")) if payload.get("age") is not None else None
        bmi = float(payload.get("bmi")) if payload.get("bmi") is not None else None
        sys_bp = float(payload.get("systolic_bp")) if payload.get("systolic_bp") is not None else None
        dia_bp = float(payload.get("diastolic_bp")) if payload.get("diastolic_bp") is not None else None
        hr = float(payload.get("heart_rate")) if payload.get("heart_rate") is not None else None
        ex_level = parse_exercise_level(payload.get("exercise_level"))
    except Exception as e:
        return jsonify({"error": "bad_request", "message": f"Invalid input types: {e}"}), 400

    # Validate required
    required = {"age": age, "bmi": bmi, "exercise_level": ex_level, "systolic_bp": sys_bp, "diastolic_bp": dia_bp, "heart_rate": hr}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        return jsonify({"error": "bad_request", "message": f"Missing/invalid fields: {missing}"}), 400

    # Predict
    if MODEL is None:
        return jsonify({"error": "model_not_loaded", "message": MODEL_ERROR or "Model not loaded"}), 500

    x = np.array([[age, bmi, ex_level, sys_bp, dia_bp, hr]], dtype=float)

    try:
        pred = int(MODEL.predict(x)[0])
        prob_high = float(MODEL.predict_proba(x)[0][1])
        risk_label = "High Risk" if pred == 1 else "Low Risk"
        probability = round(prob_high * 100.0, 2)
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": "predict_failed", "message": str(e)}), 500

    # Save (anonymous allowed: user_id None)
    try:
        row = Prediction(
            client_id=client_id,
            user_id=None,
            created_at=datetime.now(timezone.utc),
            age=age,
            bmi=bmi,
            exercise_level=ex_level,  # numeric always
            systolic_bp=sys_bp,
            diastolic_bp=dia_bp,
            heart_rate=hr,
            risk_label=risk_label,
            probability=probability,
        )
        db.session.add(row)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.exception("DB insert failed")
        return jsonify({"error": "db_insert_failed", "message": str(e)}), 500

    return jsonify({"risk_label": risk_label, "probability": probability, "client_id": client_id})


# ----------------------------
# Local dev entry
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), debug=True)
