import os
import uuid
import json
import time
import secrets
import datetime as dt
import logging

import numpy as np
import joblib

from flask import Flask, render_template, request, redirect, url_for, jsonify, g
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


# ----------------------------
# App + Config
# ----------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL or "sqlite:///local.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Basic abuse protection: cap request body (bytes)
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", "20000"))  # 20KB default
app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES

db = SQLAlchemy(app)


# ----------------------------
# Structured logging
# ----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("early-risk-alert-ai")

START_TIME = time.time()


def log_event(event: str, **fields):
    base = {
        "event": event,
        "request_id": getattr(g, "request_id", None),
        "client_id": getattr(g, "client_id", None),
        "user_id": getattr(g, "user_id", None),
        "ip": get_remote_address(),
        "path": request.path if request else None,
        "method": request.method if request else None,
    }
    base.update(fields)
    logger.info(json.dumps(base, default=str))


# ----------------------------
# Rate limiting
# ----------------------------
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[
        "500 per day",
        "120 per hour",
    ],
)


def client_key():
    return getattr(g, "client_id", get_remote_address())


def user_key():
    return getattr(g, "user_token", getattr(g, "client_id", get_remote_address()))


# ----------------------------
# Model load
# ----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "demo_model.pkl")
model = joblib.load(MODEL_PATH)


# ----------------------------
# Anonymous + Upgrade-to-account models
# ----------------------------
CLIENT_ID_COOKIE = "client_id"
CLIENT_ID_HEADER = "X-Client-Id"

USER_TOKEN_HEADER = "X-User-Token"  # mobile stores this after upgrade
API_KEY_HEADER = "X-API-Key"        # simple gate for /api/*
API_KEYS = {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}


class User(db.Model):
    __tablename__ = "user_account"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    verified = db.Column(db.Boolean, default=False)
    user_token = db.Column(db.String(64), unique=True, nullable=False)  # returned to app
    verify_code = db.Column(db.String(12), nullable=True)
    verify_expires_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)


class ClientIdentity(db.Model):
    """
    Links anonymous client_id to a user account after upgrade.
    """
    __tablename__ = "client_identity"
    client_id = db.Column(db.String(64), primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user_account.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)


class Prediction(db.Model):
    __tablename__ = "prediction"
    id = db.Column(db.Integer, primary_key=True)

    client_id = db.Column(db.String(64), index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user_account.id"), index=True, nullable=True)

    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)

    age = db.Column(db.Integer)
    bmi = db.Column(db.Float)
    exercise_level = db.Column(db.String(20))
    systolic_bp = db.Column(db.Integer)
    diastolic_bp = db.Column(db.Integer)
    heart_rate = db.Column(db.Integer)

    risk_label = db.Column(db.String(20))
    probability = db.Column(db.Float)  # stored as percent 0-100


with app.app_context():
    db.create_all()


# ----------------------------
# Helpers
# ----------------------------
def get_or_create_client_id() -> str:
    header_id = (request.headers.get(CLIENT_ID_HEADER) or "").strip()
    if header_id:
        return header_id

    cookie_id = (request.cookies.get(CLIENT_ID_COOKIE) or "").strip()
    if cookie_id:
        return cookie_id

    return str(uuid.uuid4())


def attach_user_from_token():
    token = (request.headers.get(USER_TOKEN_HEADER) or "").strip()
    if not token:
        return None, None
    user = User.query.filter_by(user_token=token, verified=True).first()
    if not user:
        return None, token
    return user, token


def ensure_client_identity_row(client_id: str):
    row = ClientIdentity.query.filter_by(client_id=client_id).first()
    if not row:
        row = ClientIdentity(client_id=client_id)
        db.session.add(row)
        db.session.commit()
    return row


def current_owner_filter():
    """
    If upgraded (user_id exists), show user history.
    Otherwise show client_id history.
    """
    if getattr(g, "user_id", None):
        return ("user_id", g.user_id)
    return ("client_id", g.client_id)


def require_api_key():
    """
    Simple abuse gate for /api/* endpoints.
    For public web, do NOT require.
    """
    if not API_KEYS:
        # If you haven't set API_KEYS yet, allow (dev mode).
        return True

    key = (request.headers.get(API_KEY_HEADER) or "").strip()
    return key in API_KEYS


def validate_inputs(age, bmi, exercise_level, sys_bp, dia_bp, heart_rate):
    # strict + sane bounds (abuse protection)
    if not (0 <= age <= 120):
        return "age out of range"
    if not (10.0 <= bmi <= 80.0):
        return "bmi out of range"
    if exercise_level not in ("Low", "Moderate", "High"):
        return "exercise_level invalid"
    if not (70 <= sys_bp <= 250):
        return "systolic_bp out of range"
    if not (40 <= dia_bp <= 150):
        return "diastolic_bp out of range"
    if not (30 <= heart_rate <= 220):
        return "heart_rate out of range"
    return None


EXERCISE_MAP = {"Low": 0, "Moderate": 1, "High": 2}


# ----------------------------
# Request lifecycle
# ----------------------------
@app.before_request
def attach_context():
    g.request_id = str(uuid.uuid4())
    g.client_id = get_or_create_client_id()

    # Ensure client exists in mapping table
    ensure_client_identity_row(g.client_id)

    # Attempt to attach user via user_token
    user, token = attach_user_from_token()
    g.user_token = token
    g.user_id = user.id if user else None

    log_event("request_start")


@app.after_request
def set_cookie_and_log(response):
    # Set cookie only for web users (no mobile header token)
    if not request.headers.get(CLIENT_ID_HEADER):
        if not request.cookies.get(CLIENT_ID_COOKIE):
            response.set_cookie(
                CLIENT_ID_COOKIE,
                g.client_id,
                max_age=60 * 60 * 24 * 365 * 2,
                httponly=True,
                secure=True,   # keep True in production (HTTPS)
                samesite="Lax",
            )

    log_event("request_end", status=response.status_code)
    return response


# ----------------------------
# Web Routes
# ----------------------------
@app.get("/")
def home():
    return render_template("index.html")


@app.post("/predict")
@limiter.limit("20/minute", key_func=client_key)
def predict():
    try:
        age = int(request.form["age"])
        bmi = float(request.form["bmi"])
        exercise = request.form["exercise_level"]
        sys_bp = int(request.form["systolic_bp"])
        dia_bp = int(request.form["diastolic_bp"])
        heart_rate = int(request.form["heart_rate"])
    except Exception:
        log_event("predict_invalid_form")
        return "Invalid input", 400

    err = validate_inputs(age, bmi, exercise, sys_bp, dia_bp, heart_rate)
    if err:
        log_event("predict_validation_failed", reason=err)
        return f"Invalid input: {err}", 400

    x = np.array([[age, bmi, EXERCISE_MAP[exercise], sys_bp, dia_bp, heart_rate]])
    pred = int(model.predict(x)[0])
    prob_high = float(model.predict_proba(x)[0][1])

    risk_label = "High Risk" if pred == 1 else "Low Risk"
    probability = round(prob_high * 100, 2)

    record = Prediction(
        client_id=g.client_id,
        user_id=g.user_id,
        age=age,
        bmi=bmi,
        exercise_level=exercise,
        systolic_bp=sys_bp,
        diastolic_bp=dia_bp,
        heart_rate=heart_rate,
        risk_label=risk_label,
        probability=probability,
    )
    db.session.add(record)
    db.session.commit()

    log_event("prediction_saved", risk_label=risk_label, probability=probability)
    return redirect(url_for("history"))


@app.get("/history")
def history():
    key, val = current_owner_filter()
    q = Prediction.query.filter(getattr(Prediction, key) == val).order_by(Prediction.id.desc()).limit(200)
    rows = q.all()
    return render_template("history.html", rows=rows)


# ----------------------------
# API Routes (mobile / external)
# ----------------------------
@app.post("/api/predict")
@limiter.limit("30/minute", key_func=user_key)  # per user_token or client_id
def api_predict():
    if not require_api_key():
        log_event("api_denied_bad_key")
        return jsonify({"error": "Unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    try:
        age = int(payload["age"])
        bmi = float(payload["bmi"])
        exercise = str(payload["exercise_level"])
        sys_bp = int(payload["systolic_bp"])
        dia_bp = int(payload["diastolic_bp"])
        heart_rate = int(payload["heart_rate"])
    except Exception:
        log_event("api_predict_invalid_json")
        return jsonify({"error": "Invalid payload"}), 400

    err = validate_inputs(age, bmi, exercise, sys_bp, dia_bp, heart_rate)
    if err:
        log_event("api_predict_validation_failed", reason=err)
        return jsonify({"error": err}), 400

    x = np.array([[age, bmi, EXERCISE_MAP[exercise], sys_bp, dia_bp, heart_rate]])
    pred = int(model.predict(x)[0])
    prob_high = float(model.predict_proba(x)[0][1])

    risk_label = "High Risk" if pred == 1 else "Low Risk"
    probability = round(prob_high * 100, 2)

    record = Prediction(
        client_id=g.client_id,
        user_id=g.user_id,
        age=age,
        bmi=bmi,
        exercise_level=exercise,
        systolic_bp=sys_bp,
        diastolic_bp=dia_bp,
        heart_rate=heart_rate,
        risk_label=risk_label,
        probability=probability,
    )
    db.session.add(record)
    db.session.commit()

    log_event("api_prediction_saved", risk_label=risk_label, probability=probability)
    return jsonify({"risk_label": risk_label, "probability": probability}), 200


# ----------------------------
# Upgrade-to-account API
# ----------------------------
@app.post("/api/upgrade/start")
@limiter.limit("5/hour", key_func=client_key)
def upgrade_start():
    if not require_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip().lower()

    if not email or "@" not in email or len(email) > 255:
        return jsonify({"error": "Invalid email"}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        user = User(
            email=email,
            verified=False,
            user_token=secrets.token_hex(24),
        )
        db.session.add(user)
        db.session.commit()

    # Create a 6-digit code (MVP). In production you email/SMS it.
    code = f"{secrets.randbelow(1000000):06d}"
    user.verify_code = code
    user.verify_expires_at = dt.datetime.utcnow() + dt.timedelta(minutes=15)
    db.session.commit()

    log_event("upgrade_started", email=email)

    # IMPORTANT:
    # For real production, DO NOT return the code.
    # Email/SMS it. Returning here is for development/testing.
    return jsonify({"status": "sent", "dev_code": code}), 200


@app.post("/api/upgrade/confirm")
@limiter.limit("10/hour", key_func=client_key)
def upgrade_confirm():
    if not require_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    code = (payload.get("code") or "").strip()

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"error": "Invalid email"}), 400

    if not user.verify_code or not user.verify_expires_at:
        return jsonify({"error": "No active code"}), 400

    if dt.datetime.utcnow() > user.verify_expires_at:
        return jsonify({"error": "Code expired"}), 400

    if code != user.verify_code:
        log_event("upgrade_confirm_failed", email=email)
        return jsonify({"error": "Invalid code"}), 400

    user.verified = True
    user.verify_code = None
    user.verify_expires_at = None
    db.session.commit()

    # Link this anonymous client_id to the user
    ident = ClientIdentity.query.filter_by(client_id=g.client_id).first()
    if not ident:
        ident = ClientIdentity(client_id=g.client_id)
        db.session.add(ident)
    ident.user_id = user.id
    db.session.commit()

    # Optionally attach existing predictions for this client_id to user_id
    Prediction.query.filter_by(client_id=g.client_id).update({"user_id": user.id})
    db.session.commit()

    log_event("upgrade_confirmed", email=email, new_user_id=user.id)
    return jsonify({"status": "upgraded", "user_token": user.user_token}), 200


# ----------------------------
# Health
# ----------------------------
@app.get("/healthz")
def healthz():
    db_ok = True
    db_error = None
    try:
        db.session.execute(text("SELECT 1"))
    except Exception as e:
        db_ok = False
        db_error = str(e)

    return (
        {
            "status": "ok" if db_ok else "degraded",
            "db_ok": db_ok,
            "db_error": db_error,
            "model_loaded": model is not None,
            "uptime_seconds": round(time.time() - START_TIME, 2),
        },
        200 if db_ok else 503,
    )


# Local run only
if __name__ == "__main__":
    app.run(debug=True)
