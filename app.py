import os
import time
import json
import secrets
import logging
from datetime import datetime, timezone
from collections import defaultdict, deque

import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template, g
from flask_sqlalchemy import SQLAlchemy

# ----------------------------
# App setup
# ----------------------------
app = Flask(__name__)

# Secret key (Flask sessions, flash, etc.)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")

# DB (Render provides DATABASE_URL)
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL or "sqlite:///local.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ----------------------------
# Logging
# ----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(message)s")
logger = logging.getLogger("early-risk-alert-ai")


def log_event(**fields):
    # structured JSON logs (Render-friendly)
    fields.setdefault("ts", datetime.now(timezone.utc).isoformat())
    logger.info(json.dumps(fields, default=str))


# ----------------------------
# Model load
# ----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "demo_model.pkl")
model = None
model_error = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model_error = f"MODEL_PATH not found: {MODEL_PATH}"
except Exception as e:
    model_error = f"Model load error: {e}"

# ----------------------------
# API key + behavior flags
# ----------------------------
# API_KEYS can be a comma-separated list: "key1,key2,key3"
API_KEYS_RAW = os.getenv("API_KEYS", "").strip()
API_KEYS = {k.strip() for k in API_KEYS_RAW.split(",") if k.strip()}

REQUIRE_API_KEY_FOR_API = os.getenv("REQUIRE_API_KEY_FOR_API", "true").lower() in (
    "1",
    "true",
    "yes",
    "y",
)
ALLOW_ANONYMOUS_SAVES = os.getenv("ALLOW_ANONYMOUS_SAVES", "true").lower() in (
    "1",
    "true",
    "yes",
    "y",
)
SAVE_PREDICTIONS = os.getenv("SAVE_PREDICTIONS", "true").lower() in (
    "1",
    "true",
    "yes",
    "y",
)

# Rate limiting (simple in-memory token bucket-ish window)
RATE_LIMIT_PER_MIN = int(
    os.getenv("RATE_LIMIT_PER_MIN", "60")
)  # requests per minute per client
_rl_window = defaultdict(lambda: deque())  # client_id -> deque[timestamps]

# Basic metrics
_metrics = {
    "requests_total": 0,
    "predict_total": 0,
    "predict_4xx": 0,
    "predict_5xx": 0,
    "last_predict_ms": 0,
    "avg_predict_ms": 0,
}
_predict_times = deque(maxlen=200)  # rolling window

# ----------------------------
# DB Models
# ----------------------------
class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    client_id = db.Column(db.String(120), nullable=True)
    age = db.Column(db.Float, nullable=True)
    bmi = db.Column(db.Float, nullable=True)
    exercise_level = db.Column(db.String(32), nullable=True)

    # Keep both names to match older schema versions
    systolic_bp = db.Column(db.Float, nullable=True)
    sys_bp = db.Column(db.Float, nullable=True)

    diastolic_bp = db.Column(db.Float, nullable=True)
    dia_bp = db.Column(db.Float, nullable=True)

    heart_rate = db.Column(db.Float, nullable=True)

    risk_label = db.Column(db.String(32), nullable=True)
    probability = db.Column(db.Float, nullable=True)


# Optional: request log table (lightweight analytics)
class ApiRequestLog(db.Model):
    __tablename__ = "api_request_log"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(
        db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    request_id = db.Column(db.String(64), nullable=True)
    client_id = db.Column(db.String(120), nullable=True)
    path = db.Column(db.String(200), nullable=True)
    method = db.Column(db.String(16), nullable=True)
    status = db.Column(db.Integer, nullable=True)
    duration_ms = db.Column(db.Integer, nullable=True)
    ip = db.Column(db.String(64), nullable=True)


with app.app_context():
    db.create_all()

# ----------------------------
# Helpers
# ----------------------------
def normalize_exercise_level(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        # allow numeric encoding if user sends it
        if x <= 0:
            return "Low"
        if x == 1:
            return "Moderate"
        return "High"
    s = str(x).strip().lower()
    if s in ("low", "l", "0"):
        return "Low"
    if s in ("moderate", "mod", "m", "1", "medium"):
        return "Moderate"
    if s in ("high", "h", "2"):
        return "High"
    return str(x).strip()


def exercise_to_numeric(level_str):
    # match your training pipeline expectations (0,1,2)
    lvl = (level_str or "").strip().lower()
    if lvl == "low":
        return 0.0
    if lvl == "moderate":
        return 1.0
    if lvl == "high":
        return 2.0
    # unknown -> treat as moderate
    return 1.0


def get_client_id():
    # Prefer header; fallback to "anonymous"
    return (request.headers.get("X-Client-Id") or "").strip() or "anonymous"


def require_api_key_if_enabled():
    if not REQUIRE_API_KEY_FOR_API:
        return None
    provided = (request.headers.get("X-API-Key") or "").strip()
    if not provided:
        return ("auth_required", "X-API-Key header required"), 401
    if API_KEYS and provided not in API_KEYS:
        return ("auth_invalid", "Invalid API key"), 403
    return None


def rate_limit_check(client_id):
    now = time.time()
    window = _rl_window[client_id]
    # drop timestamps older than 60s
    while window and (now - window[0]) > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MIN:
        return True
    window.append(now)
    return False


def json_error(code, message, status=400, **extra):
    payload = {"error": code, "message": message, **extra}
    return jsonify(payload), status


def record_metrics_predict(ms, status_code):
    _metrics["predict_total"] += 1
    _metrics["last_predict_ms"] = int(ms)
    _predict_times.append(ms)
    _metrics["avg_predict_ms"] = int(sum(_predict_times) / len(_predict_times))

    if 400 <= status_code < 500:
        _metrics["predict_4xx"] += 1
    if status_code >= 500:
        _metrics["predict_5xx"] += 1


def save_api_request_log(status, duration_ms):
    try:
        log = ApiRequestLog(
            request_id=getattr(g, "request_id", None),
            client_id=get_client_id(),
            path=request.path,
            method=request.method,
            status=status,
            duration_ms=int(duration_ms),
            ip=request.headers.get("X-Forwarded-For", request.remote_addr),
        )
        db.session.add(log)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        log_event(level="WARN", msg="api_request_log_failed", err=str(e))


# ----------------------------
# Request lifecycle hooks
# ----------------------------
@app.before_request
def _before():
    g.request_start = time.time()
    g.request_id = request.headers.get("X-Request-Id") or secrets.token_hex(8)
    _metrics["requests_total"] += 1


@app.after_request
def _after(resp):
    # security headers (basic hardening)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

    # request logging
    dur_ms = (time.time() - getattr(g, "request_start", time.time())) * 1000
    log_event(
        level="INFO",
        msg="request_end",
        request_id=getattr(g, "request_id", None),
        method=request.method,
        path=request.path,
        status=resp.status_code,
        duration_ms=int(dur_ms),
        client_id=get_client_id(),
    )

    # lightweight analytics
    if request.path.startswith("/api/"):
        save_api_request_log(resp.status_code, dur_ms)

    resp.headers["X-Request-Id"] = getattr(g, "request_id", "")
    return resp


# ----------------------------
# Routes (UI optional)
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    # If you have templates, keep using them. Otherwise show a simple message.
    try:
        return render_template("index.html")
    except Exception:
        return "Early Risk Alert AI is running. Use /healthz and /api/v1/predict", 200


# ----------------------------
# Health + Metrics
# ----------------------------
@app.route("/healthz", methods=["GET"])
def healthz():
    db_ok = True
    db_error = None
    try:
        # simple query to validate DB connection
        db.session.execute(db.text("SELECT 1"))
    except Exception as e:
        db_ok = False
        db_error = str(e)

    return (
        jsonify(
            {
                "status": "ok" if (db_ok and model is not None) else "degraded",
                "model_loaded": model is not None,
                "model_error": model_error,
                "db_ok": db_ok,
                "db_error": db_error,
                "uptime_seconds": int(
                    time.time() - getattr(app, "_started_at", time.time())
                ),
                "version": "v1",
            }
        ),
        200,
    )


@app.route("/metrics", methods=["GET"])
def metrics():
    # basic JSON metrics (good enough for Render logs / external checks)
    return (
        jsonify(
            {
                **_metrics,
                "rate_limit_per_min": RATE_LIMIT_PER_MIN,
                "require_api_key_for_api": REQUIRE_API_KEY_FOR_API,
                "model_loaded": model is not None,
                "db_using": "postgres" if DATABASE_URL else "sqlite",
            }
        ),
        200,
    )


app._started_at = time.time()

# ----------------------------
# Predict API (v1) + backward compatible route
# ----------------------------
def _predict_impl():
    # auth
    auth_err = require_api_key_if_enabled()
    if auth_err:
        return json_error(auth_err[0][0], auth_err[0][1], auth_err[1])

    client_id = get_client_id()

    # rate limit
    if rate_limit_check(client_id):
        return json_error("rate_limited", "Too many requests. Try again in a minute.", 429)

    # model available?
    if model is None:
        return json_error(
            "model_unavailable",
            "Model is not loaded on the server.",
            503,
            detail=model_error,
        )

    # parse JSON
    payload = request.get_json(silent=True) or {}
    try:
        age = float(payload.get("age"))
        bmi = float(payload.get("bmi"))
        exercise_level = normalize_exercise_level(payload.get("exercise_level"))
        systolic_bp = float(payload.get("systolic_bp"))
        diastolic_bp = float(payload.get("diastolic_bp"))
        heart_rate = float(payload.get("heart_rate"))
    except Exception:
        return json_error(
            "bad_request",
            "Expected numeric: age,bmi,systolic_bp,diastolic_bp,heart_rate and string: exercise_level",
            400,
        )

    # feature vector (match your training order)
    ex_num = exercise_to_numeric(exercise_level)
    X = np.array([[age, bmi, ex_num, systolic_bp, diastolic_bp, heart_rate]], dtype=float)

    start = time.time()
    try:
        pred = int(model.predict(X)[0])
        prob_high = float(model.predict_proba(X)[0][1])
        risk_label = "High Risk" if pred == 1 else "Low Risk"
        probability = round(prob_high * 100, 2)

        # save prediction (optional)
        if SAVE_PREDICTIONS and (ALLOW_ANONYMOUS_SAVES or client_id != "anonymous"):
            p = Prediction(
                client_id=client_id,
                age=age,
                bmi=bmi,
                exercise_level=exercise_level,
                systolic_bp=systolic_bp,
                sys_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                dia_bp=diastolic_bp,
                heart_rate=heart_rate,
                risk_label=risk_label,
                probability=probability,
            )
            db.session.add(p)
            db.session.commit()

        ms = (time.time() - start) * 1000
        record_metrics_predict(ms, 200)

        return (
            jsonify(
                {
                    "client_id": client_id,
                    "probability": probability,
                    "risk_label": risk_label,
                }
            ),
            200,
        )

    except Exception as e:
        db.session.rollback()
        ms = (time.time() - start) * 1000
        record_metrics_predict(ms, 500)
        log_event(level="ERROR", msg="predict_failed", err=str(e), client_id=client_id)
        return json_error("server_error", "Prediction failed.", 500)


@app.route("/api/v1/predict", methods=["POST"])
def api_v1_predict():
    return _predict_impl()


# Backward compatible route (keeps your old curl commands working)
@app.route("/api/predict", methods=["POST"])
def api_predict_legacy():
    return _predict_impl()


# ----------------------------
# Friendly JSON 404/405 for API
# ----------------------------
@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return json_error("not_found", "API route not found.", 404)
    return "Not Found", 404


@app.errorhandler(405)
def method_not_allowed(e):
    if request.path.startswith("/api/"):
        return json_error("method_not_allowed", "Method not allowed for this endpoint.", 405)
    return "Method Not Allowed", 405


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
