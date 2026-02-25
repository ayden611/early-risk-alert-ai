import os
import json
import time
import uuid
import hmac
import base64
import hashlib
import logging
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np

from flask import Flask, request, jsonify, g
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# ------------------------------------
# Config
# ------------------------------------
APP_NAME = "early-risk-alert-ai"
UTC = timezone.utc

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///local.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
JWT_SECRET = os.getenv("JWT_SECRET", SECRET_KEY)
JWT_ISSUER = os.getenv("JWT_ISSUER", APP_NAME)

# Model artifacts
MODEL_PATH = os.getenv("MODEL_PATH", "demo_model.pkl")
MODEL_META_PATH = os.getenv("MODEL_META_PATH", "model_meta.json")

# Abuse protection
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", "32768"))  # 32KB
REQUIRE_CLIENT_ID = os.getenv("REQUIRE_CLIENT_ID", "true").lower() == "true"
REQUIRE_API_KEY_FOR_API = os.getenv("REQUIRE_API_KEY_FOR_API", "true").lower() == "true"

# Rate limits (simple in-memory; production multi-instance should use Redis)
# defaults: per minute
RL_IP_PER_MIN = int(os.getenv("RL_IP_PER_MIN", "60"))            # 60 req/min per IP
RL_CLIENT_PER_MIN = int(os.getenv("RL_CLIENT_PER_MIN", "120"))   # 120 req/min per client
RL_KEY_PER_MIN = int(os.getenv("RL_KEY_PER_MIN", "240"))         # 240 req/min per API key

# CORS (if you need it later for mobile; keep off by default)
ENABLE_CORS = os.getenv("ENABLE_CORS", "false").lower() == "true"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# ------------------------------------
# App + DB
# ------------------------------------
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JSON_SORT_KEYS"] = False
app.secret_key = SECRET_KEY
db = SQLAlchemy(app)

# ------------------------------------
# Structured logging (JSON)
# ------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(APP_NAME)
logger.setLevel(LOG_LEVEL)
handler = logging.StreamHandler()
handler.setLevel(LOG_LEVEL)


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "service": APP_NAME,
            "msg": record.getMessage(),
        }
        # Attach request context if available
        rid = getattr(g, "request_id", None)
        if rid:
            payload["request_id"] = rid
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload, ensure_ascii=False)


handler.setFormatter(JsonFormatter())
logger.handlers = [handler]

# ------------------------------------
# Helpers
# ------------------------------------
def now_utc() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat()


def get_client_ip() -> str:
    # Render/Cloudflare often sets these; fall back safely
    hdr = (
        request.headers.get("CF-Connecting-IP")
        or request.headers.get("X-Forwarded-For")
        or request.remote_addr
    )
    if hdr and "," in hdr:
        hdr = hdr.split(",")[0].strip()
    return hdr or "unknown"


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    # prefix for debugging + long random token
    prefix = "era"
    token = base64.urlsafe_b64encode(os.urandom(32)).decode("utf-8").rstrip("=")
    return f"{prefix}_{token}"


def safe_str(v, max_len=200):
    if v is None:
        return None
    s = str(v)
    return s[:max_len]


def json_error(status: int, code: str, message: str, **extra):
    payload = {"error": code, "message": message}
    if extra:
        payload.update(extra)
    return jsonify(payload), status


# Minimal JWT (no extra dependency)
def jwt_sign(payload: dict, expires_minutes: int = 60) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    exp = int((now_utc() + timedelta(minutes=expires_minutes)).timestamp())
    body = {**payload, "iss": JWT_ISSUER, "exp": exp}

    def b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")

    header_b = b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    body_b = b64url(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b}.{body_b}".encode("utf-8")
    sig = hmac.new(JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b = b64url(sig)
    return f"{header_b}.{body_b}.{sig_b}"


def jwt_verify(token: str) -> dict | None:
    try:
        header_b, body_b, sig_b = token.split(".")
        signing_input = f"{header_b}.{body_b}".encode("utf-8")

        def b64url_decode(s: str) -> bytes:
            pad = "=" * (-len(s) % 4)
            return base64.urlsafe_b64decode(s + pad)

        sig = b64url_decode(sig_b)
        expected = hmac.new(
            JWT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        if not hmac.compare_digest(sig, expected):
            return None

        body = json.loads(b64url_decode(body_b))
        if body.get("iss") != JWT_ISSUER:
            return None
        if int(body.get("exp", 0)) < int(now_utc().timestamp()):
            return None
        return body
    except Exception:
        return None


# ------------------------------------
# Rate limiting (in-memory)
# ------------------------------------
# NOTE: For true production multi-instance (Render scaling), move this to Redis.
_rl = {
    "ip": {},  # key -> [timestamps]
    "client": {},  # key -> [timestamps]
    "apikey": {},  # key -> [timestamps]
}


def _rl_hit(bucket: str, key: str, limit_per_min: int) -> bool:
    # returns True if allowed, False if rate-limited
    if not key:
        return True
    now = time.time()
    window = 60.0
    arr = _rl[bucket].get(key, [])
    arr = [t for t in arr if (now - t) < window]
    if len(arr) >= limit_per_min:
        _rl[bucket][key] = arr
        return False
    arr.append(now)
    _rl[bucket][key] = arr
    return True


# ------------------------------------
# DB Models
# ------------------------------------
class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=now_utc, nullable=False)


class ApiKey(db.Model):
    __tablename__ = "api_key"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)
    name = db.Column(db.String(120), nullable=True)
    key_prefix = db.Column(db.String(16), nullable=False, index=True)
    key_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=now_utc, nullable=False)
    last_used_at = db.Column(db.DateTime(timezone=True), nullable=True)


class Prediction(db.Model):
    __tablename__ = "prediction"
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime(timezone=True), default=now_utc, nullable=False)

    # Tracking
    request_id = db.Column(db.String(64), nullable=True, index=True)
    client_id = db.Column(db.String(80), nullable=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)

    # Inputs
    age = db.Column(db.Float, nullable=True)
    bmi = db.Column(db.Float, nullable=True)
    exercise_level = db.Column(db.String(32), nullable=True)  # store human label
    systolic_bp = db.Column(db.Float, nullable=True)
    diastolic_bp = db.Column(db.Float, nullable=True)
    heart_rate = db.Column(db.Float, nullable=True)

    # Outputs
    risk_label = db.Column(db.String(32), nullable=False)
    probability = db.Column(db.Float, nullable=False)


class UsageDaily(db.Model):
    __tablename__ = "usage_daily"
    id = db.Column(db.Integer, primary_key=True)
    day = db.Column(db.String(10), nullable=False, index=True)  # YYYY-MM-DD
    api_key_id = db.Column(
        db.Integer, db.ForeignKey("api_key.id"), nullable=True, index=True
    )
    client_id = db.Column(db.String(80), nullable=True, index=True)
    count = db.Column(db.Integer, default=0, nullable=False)


# ------------------------------------
# Model loading
# ------------------------------------
model = None
model_loaded = False
model_error = None


def load_model():
    global model, model_loaded, model_error
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            model_loaded = True
            model_error = None
        else:
            model = None
            model_loaded = False
            model_error = f"MODEL_PATH not found: {MODEL_PATH}"
    except Exception as e:
        model = None
        model_loaded = False
        model_error = str(e)


def map_exercise_level(ex: str) -> float | str:
    # try to satisfy either numeric or string-based models
    if ex is None:
        return "Low"
    ex_s = str(ex).strip().lower()
    if ex_s in ("low", "0"):
        return 0.0
    if ex_s in ("moderate", "medium", "1"):
        return 1.0
    if ex_s in ("high", "2"):
        return 2.0
    # default keep string (some pipelines handle categories)
    return str(ex).title()


def coerce_float(v, field):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        raise ValueError(f"Invalid number for '{field}'")


def predict_payload_to_features(data: dict) -> tuple[np.ndarray, dict]:
    age = coerce_float(data.get("age"), "age")
    bmi = coerce_float(data.get("bmi"), "bmi")
    systolic_bp = coerce_float(data.get("systolic_bp"), "systolic_bp")
    diastolic_bp = coerce_float(data.get("diastolic_bp"), "diastolic_bp")
    heart_rate = coerce_float(data.get("heart_rate"), "heart_rate")

    ex_raw = data.get("exercise_level", "Low")
    ex_mapped = map_exercise_level(ex_raw)

    # Build feature array (common order used in your app)
    # If your model pipeline expects different columns, adjust here once.
    X = np.array(
        [
            [
                age,
                bmi,
                float(ex_mapped) if isinstance(ex_mapped, float) else 0.0,
                systolic_bp,
                diastolic_bp,
                heart_rate,
            ]
        ]
    )

    saved = {
        "age": age,
        "bmi": bmi,
        "exercise_level": str(ex_raw).title() if ex_raw is not None else None,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "heart_rate": heart_rate,
    }
    return X, saved


# ------------------------------------
# Startup: DB + model
# ------------------------------------
with app.app_context():
    db.create_all()
    load_model()

# ------------------------------------
# Middleware
# ------------------------------------
@app.before_request
def _before():
    g.request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    g.start = time.time()
    g.ip = get_client_ip()

    # basic payload abuse protection
    cl = request.content_length
    if cl is not None and cl > MAX_BODY_BYTES:
        return json_error(
            413, "payload_too_large", f"Max body size is {MAX_BODY_BYTES} bytes"
        )

    # rate limit per IP (all endpoints)
    if not _rl_hit("ip", g.ip, RL_IP_PER_MIN):
        return json_error(429, "rate_limited", "Too many requests (IP).")


@app.after_request
def _after(resp):
    duration_ms = int((time.time() - getattr(g, "start", time.time())) * 1000)
    resp.headers["X-Request-Id"] = getattr(g, "request_id", "")
    if ENABLE_CORS:
        resp.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGINS
        resp.headers[
            "Access-Control-Allow-Headers"
        ] = "Content-Type, Authorization, X-API-Key, X-Client-Id, X-Request-Id"
        resp.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS"
    # structured log
    try:
        logger.info(
            "request_end",
            extra={
                "extra": {
                    "event": "request_end",
                    "method": request.method,
                    "path": request.path,
                    "status": resp.status_code,
                    "duration_ms": duration_ms,
                    "ip": getattr(g, "ip", None),
                    "client_id": getattr(g, "client_id", None),
                    "user_id": getattr(g, "user_id", None),
                    "api_key_id": getattr(g, "api_key_id", None),
                }
            },
        )
    except Exception:
        pass
    return resp


@app.route("/healthz", methods=["GET"])
def healthz():
    db_ok = True
    db_error = None
    try:
        db.session.execute(db.text("SELECT 1"))
    except Exception as e:
        db_ok = False
        db_error = str(e)
    return jsonify(
        {
            "status": "ok" if (db_ok and model_loaded) else "degraded",
            "db_ok": db_ok,
            "db_error": db_error,
            "model_loaded": model_loaded,
            "model_error": model_error,
            "uptime_seconds": 0,
        }
    )

# ------------------------------------
# Auth: users + JWT
# ------------------------------------
def require_bearer_user() -> User | None:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth.split(" ", 1)[1].strip()
    body = jwt_verify(token)
    if not body:
        return None
    uid = body.get("sub")
    if not uid:
        return None
    user = User.query.get(int(uid))
    return user


@app.route("/auth/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or "@" not in email:
        return json_error(400, "invalid_email", "Valid email required.")
    if len(password) < 8:
        return json_error(400, "weak_password", "Password must be at least 8 characters.")

    if User.query.filter_by(email=email).first():
        return json_error(409, "email_taken", "Email already registered.")

    user = User(email=email, password_hash=generate_password_hash(password))
    db.session.add(user)
    db.session.commit()

    token = jwt_sign({"sub": str(user.id)}, expires_minutes=60 * 24 * 7)  # 7 days
    return jsonify({"token": token, "user": {"id": user.id, "email": user.email}}), 201


@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return json_error(401, "invalid_credentials", "Email/password incorrect.")
    token = jwt_sign({"sub": str(user.id)}, expires_minutes=60 * 24 * 7)
    return jsonify({"token": token, "user": {"id": user.id, "email": user.email}})


# ------------------------------------
# API Key auth
# ------------------------------------
def resolve_api_key() -> ApiKey | None:
    raw = (request.headers.get("X-API-Key") or "").strip()
    if not raw:
        return None
    key_hash = sha256_hex(raw)
    k = ApiKey.query.filter_by(key_hash=key_hash, is_active=True).first()
    return k


def require_api_auth():
    # attach client_id / api_key / user to g
    g.client_id = safe_str(request.headers.get("X-Client-Id"), 80)
    if REQUIRE_CLIENT_ID and not g.client_id:
        return json_error(400, "missing_client_id", "X-Client-Id header required.")

    # Rate limit per client
    if g.client_id and not _rl_hit("client", g.client_id, RL_CLIENT_PER_MIN):
        return json_error(429, "rate_limited", "Too many requests (client).")

    # API key (recommended for mobile apps)
    if REQUIRE_API_KEY_FOR_API:
        k = resolve_api_key()
        if not k:
            return json_error(401, "missing_api_key", "X-API-Key required.")
        g.api_key_id = k.id
        # rate limit per key
        if not _rl_hit("apikey", str(k.id), RL_KEY_PER_MIN):
            return json_error(429, "rate_limited", "Too many requests (api key).")
        k.last_used_at = now_utc()
        db.session.commit()
    else:
        g.api_key_id = None

    # User is optional (JWT). If present, attach.
    u = require_bearer_user()
    g.user_id = u.id if u else None
    return None


@app.route("/keys/create", methods=["POST"])
def create_key():
    # Must be logged in (JWT) to create a key
    user = require_bearer_user()
    if not user:
        return json_error(401, "auth_required", "Bearer token required.")

    data = request.get_json(silent=True) or {}
    name = safe_str(data.get("name"), 120) or "default"

    raw = generate_api_key()
    key_hash = sha256_hex(raw)
    prefix = raw.split("_", 1)[0] + "_" + raw.split("_", 1)[1][:6]

    k = ApiKey(user_id=user.id, name=name, key_prefix=prefix, key_hash=key_hash, is_active=True)
    db.session.add(k)
    db.session.commit()

    # Return raw key ONCE
    return jsonify({"api_key": raw, "key": {"id": k.id, "name": k.name, "prefix": k.key_prefix}}), 201


@app.route("/keys/revoke/<int:key_id>", methods=["DELETE"])
def revoke_key(key_id: int):
    user = require_bearer_user()
    if not user:
        return json_error(401, "auth_required", "Bearer token required.")
    k = ApiKey.query.filter_by(id=key_id, user_id=user.id).first()
    if not k:
        return json_error(404, "not_found", "Key not found.")
    k.is_active = False
    db.session.commit()
    return jsonify({"ok": True})


# ------------------------------------
# Usage tracking (billing-ready)
# ------------------------------------
def bump_usage(api_key_id: int | None, client_id: str | None):
    day = now_utc().strftime("%Y-%m-%d")
    q = UsageDaily.query.filter_by(day=day, api_key_id=api_key_id, client_id=client_id).first()
    if not q:
        q = UsageDaily(day=day, api_key_id=api_key_id, client_id=client_id, count=0)
        db.session.add(q)
    q.count += 1
    db.session.commit()


# ------------------------------------
# API: predict + history
# ------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    auth_resp = require_api_auth()
    if auth_resp:
        return auth_resp

    if not model_loaded or model is None:
        return json_error(503, "model_unavailable", "Model not loaded.", model_error=model_error)

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return json_error(400, "invalid_json", "JSON body required.")

    try:
        X, saved = predict_payload_to_features(data)
    except ValueError as e:
        return json_error(400, "invalid_input", str(e))

    try:
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else float(pred)
        risk_label = "High Risk" if pred == 1 else "Low Risk"
        probability = round(proba * 100.0, 2)
    except Exception as e:
        logger.error("predict_failed", extra={"extra": {"event": "predict_failed", "err": str(e)}})
        return json_error(500, "predict_failed", "Prediction failed.")

    # Store in DB
    p = Prediction(
        request_id=g.request_id,
        client_id=g.client_id,
        user_id=g.user_id,
        age=saved["age"],
        bmi=saved["bmi"],
        exercise_level=saved["exercise_level"],
        systolic_bp=saved["systolic_bp"],
        diastolic_bp=saved["diastolic_bp"],
        heart_rate=saved["heart_rate"],
        risk_label=risk_label,
        probability=probability,
    )
    db.session.add(p)
    db.session.commit()

    bump_usage(g.api_key_id, g.client_id)

    return jsonify({"risk_label": risk_label, "probability": probability})


@app.route("/api/history", methods=["GET"])
def api_history():
    auth_resp = require_api_auth()
    if auth_resp:
        return auth_resp

    limit = min(int(request.args.get("limit", "50")), 200)

    q = Prediction.query
    # If JWT user is present, show their history; else show by client_id
    if g.user_id:
        q = q.filter(Prediction.user_id == g.user_id)
    else:
        q = q.filter(Prediction.client_id == g.client_id)

    rows = q.order_by(Prediction.id.desc()).limit(limit).all()
    out = []
    for r in rows:
        out.append(
            {
                "id": r.id,
                "created_at": iso(r.created_at),
                "client_id": r.client_id,
                "user_id": r.user_id,
                "age": r.age,
                "bmi": r.bmi,
                "exercise_level": r.exercise_level,
                "systolic_bp": r.systolic_bp,
                "diastolic_bp": r.diastolic_bp,
                "heart_rate": r.heart_rate,
                "risk_label": r.risk_label,
                "probability": r.probability,
                "request_id": r.request_id,
            }
        )
    return jsonify({"items": out, "count": len(out)})


# ------------------------------------
# Admin metrics (minimal)
# ------------------------------------
@app.route("/admin/metrics", methods=["GET"])
def admin_metrics():
    # protect with a simple shared secret (better: admin user + role)
    token = request.headers.get("X-Admin-Token") or ""
    admin_secret = os.getenv("ADMIN_TOKEN", "")
    if not admin_secret or token != admin_secret:
        return json_error(401, "unauthorized", "Admin token required.")

    pred_count = db.session.execute(db.text("SELECT COUNT(*) FROM prediction")).scalar() or 0
    user_count = db.session.execute(db.text('SELECT COUNT(*) FROM "user"')).scalar() or 0
    key_count = db.session.execute(db.text("SELECT COUNT(*) FROM api_key")).scalar() or 0

    # last 7 days usage
    days = []
    for i in range(7):
        d = (now_utc() - timedelta(days=i)).strftime("%Y-%m-%d")
        c = (
            UsageDaily.query.filter_by(day=d)
            .with_entities(db.func.sum(UsageDaily.count))
            .scalar()
            or 0
        )
        days.append({"day": d, "requests": int(c)})
    days.reverse()

    return jsonify(
        {
            "users": int(user_count),
            "api_keys": int(key_count),
            "predictions": int(pred_count),
            "usage_last_7_days": days,
        }
    )


# ------------------------------------
# CORS preflight
# ------------------------------------
@app.route("/<path:_any>", methods=["OPTIONS"])
def _options(_any):
    return ("", 204)


# ------------------------------------
# Run local
# ------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port, debug=True)
