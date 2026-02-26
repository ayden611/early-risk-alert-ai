import os
import re
import time
import hmac
import json
import math
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text


# ============================================================
# App setup
# ============================================================
app = Flask(__name__)

def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def utc_iso(dt: datetime | None) -> str | None:
    if not dt:
        return None
    return dt.astimezone(timezone.utc).isoformat()

# Secret key for signing (required for auth tokens)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-change-me")

# DB URL normalization (Render postgres:// -> postgresql://)
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL or "sqlite:///local.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ============================================================
# Feature flags / config
# ============================================================
REQUIRE_API_KEY_FOR_API = env_bool("REQUIRE_API_KEY_FOR_API", False)
ENABLE_AUTH = env_bool("ENABLE_AUTH", True)  # OTP flow
ENABLE_RATE_LIMIT = env_bool("ENABLE_RATE_LIMIT", True)

# API keys: comma-separated list in env API_KEYS
API_KEYS = [k.strip() for k in (os.getenv("API_KEYS", "")).split(",") if k.strip()]

# Rate limit per "identity" per minute
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))

# OTP
OTP_TTL_MIN = int(os.getenv("OTP_TTL_MIN", "10"))
OTP_MAX_ATTEMPTS = int(os.getenv("OTP_MAX_ATTEMPTS", "8"))
# If no SMTP, OTP will be logged; if this is true, also returns code in response (DEV ONLY)
ALLOW_DEV_CODE_RETURN = env_bool("ALLOW_DEV_CODE_RETURN", False)

# Token TTL (bearer token)
TOKEN_TTL_DAYS = int(os.getenv("TOKEN_TTL_DAYS", "30"))

# Admin (optional)
ENABLE_ADMIN = env_bool("ENABLE_ADMIN", True)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

# Model
MODEL_PATH = os.getenv("MODEL_PATH", "demo_model.pkl")


# ============================================================
# DB Models
# ============================================================
class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=now_utc)
    last_login_at = db.Column(db.DateTime(timezone=True), nullable=True)
    is_admin = db.Column(db.Boolean, nullable=False, default=False)

class LoginCode(db.Model):
    __tablename__ = "login_code"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    code_hash = db.Column(db.String(64), nullable=False)  # sha256 hex
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=now_utc)
    expires_at = db.Column(db.DateTime(timezone=True), nullable=False)
    used_at = db.Column(db.DateTime(timezone=True), nullable=True)
    attempts = db.Column(db.Integer, nullable=False, default=0)

class Prediction(db.Model):
    __tablename__ = "prediction"
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=now_utc)

    # requester identity
    client_id = db.Column(db.String(120), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)

    # inputs
    age = db.Column(db.Float, nullable=True)
    bmi = db.Column(db.Float, nullable=True)
    exercise_level = db.Column(db.Float, nullable=True)
    systolic_bp = db.Column(db.Float, nullable=True)
    diastolic_bp = db.Column(db.Float, nullable=True)
    heart_rate = db.Column(db.Float, nullable=True)

    # outputs
    risk_label = db.Column(db.String(40), nullable=False, default="Unknown")
    probability = db.Column(db.Float, nullable=True)


with app.app_context():
    db.create_all()


# ============================================================
# Model load
# ============================================================
model = None
model_loaded = False
model_error = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        model_loaded = True
    else:
        model_error = f"MODEL_PATH not found: {MODEL_PATH}"
except Exception as e:
    model_error = f"Model load error: {e}"


# ============================================================
# Helpers: parsing, normalization, security
# ============================================================
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def normalize_email(email: str) -> str:
    return (email or "").strip().lower()

def hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()

def safe_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)

def normalize_exercise_level(x) -> float | None:
    """
    Accepts:
      - "Low"/"Moderate"/"High" (any case)
      - 0/1/2
      - numeric strings
    Returns float in {0,1,2} or None.
    """
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        # clamp-ish for safety
        v = float(x)
        if math.isnan(v):
            return None
        if v <= 0:
            return 0.0
        if v >= 2:
            return 2.0
        # if 0..2
        return float(round(v))
    s = str(x).strip().lower()
    if s in ("low", "l"):
        return 0.0
    if s in ("moderate", "mod", "m", "medium", "med"):
        return 1.0
    if s in ("high", "h"):
        return 2.0
    # numeric string?
    try:
        v = float(s)
        if math.isnan(v):
            return None
        if v <= 0:
            return 0.0
        if v >= 2:
            return 2.0
        return float(round(v))
    except Exception:
        return None

def to_float(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

def json_error(message: str, status: int = 400, **extra):
    payload = {"ok": False, "error": message}
    if extra:
        payload.update(extra)
    return jsonify(payload), status


# ============================================================
# Rate limiting (simple in-memory)
# NOTE: On multi-worker deployments this is "best effort".
# ============================================================
_rl_bucket: dict[str, list[float]] = {}

def _rl_identity() -> str:
    # Prefer user id, then client id, then api key, then ip
    auth = request.headers.get("Authorization", "")
    client_id = request.headers.get("X-Client-Id", "") or ""
    api_key = request.headers.get("X-API-Key", "") or ""
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "")
    return f"auth={auth[:24]}|client={client_id}|key={api_key[:12]}|ip={ip}"

def rate_limit_check():
    if not ENABLE_RATE_LIMIT:
        return None
    ident = _rl_identity()
    now = time.time()
    window_start = now - 60.0
    arr = _rl_bucket.get(ident, [])
    # keep only last 60s
    arr = [t for t in arr if t >= window_start]
    if len(arr) >= RATE_LIMIT_PER_MIN:
        return json_error("Rate limit exceeded", 429, rate_limit_per_min=RATE_LIMIT_PER_MIN)
    arr.append(now)
    _rl_bucket[ident] = arr
    return None


# ============================================================
# API Key gate
# ============================================================
def api_key_check():
    if not REQUIRE_API_KEY_FOR_API:
        return None
    if not API_KEYS:
        # Misconfigured: gate enabled but no keys
        return json_error("Server misconfigured: API_KEYS not set", 500)
    provided = (request.headers.get("X-API-Key") or "").strip()
    if not provided:
        return json_error("Missing X-API-Key header", 401)
    if provided not in API_KEYS:
        return json_error("Invalid API key", 401)
    return None


# ============================================================
# Token (signed) - lightweight "JWT-like" bearer token
# Format: base64url(payload_json).hexsig
# sig = HMAC_SHA256(secret, payload_bytes)
# ============================================================
def b64url_encode(b: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")

def b64url_decode(s: str) -> bytes:
    import base64
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)

def sign_token(payload: dict) -> str:
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(app.secret_key.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    return f"{b64url_encode(payload_bytes)}.{sig}"

def verify_token(token: str) -> dict | None:
    try:
        b64, sig = token.split(".", 1)
        payload_bytes = b64url_decode(b64)
        expected = hmac.new(app.secret_key.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
        if not safe_eq(expected, sig):
            return None
        payload = json.loads(payload_bytes.decode("utf-8"))
        # exp check
        exp = payload.get("exp")
        if exp is not None and time.time() > float(exp):
            return None
        return payload
    except Exception:
        return None

def current_user():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    token = auth.replace("Bearer ", "", 1).strip()
    payload = verify_token(token)
    if not payload:
        return None
    uid = payload.get("uid")
    if not uid:
        return None
    return db.session.get(User, int(uid))

def auth_check(required: bool = False):
    if not ENABLE_AUTH:
        return None, None  # auth disabled
    u = current_user()
    if required and not u:
        return json_error("Missing/invalid Bearer token", 401), None
    return None, u


# ============================================================
# Optional email sending (SMTP)
# If SMTP not set, OTP is logged.
# ============================================================
def send_email(to_email: str, subject: str, body: str) -> bool:
    """
    Configure these env vars to send:
      SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM
    If missing, returns False (no send).
    """
    host = os.getenv("SMTP_HOST")
    port = os.getenv("SMTP_PORT")
    username = os.getenv("SMTP_USERNAME")
    password = os.getenv("SMTP_PASSWORD")
    from_addr = os.getenv("SMTP_FROM")
    if not (host and port and username and password and from_addr):
        return False

    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(host, int(port)) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
        return True
    except Exception as e:
        app.logger.exception(f"SMTP send failed: {e}")
        return False


# ============================================================
# Routes
# ============================================================
@app.route("/", methods=["GET"])
def root():
    return "Early Risk Alert AI is running. Use /healthz and /api/v1/predict", 200


@app.route("/healthz", methods=["GET"])
def healthz():
    # DB check
    db_ok = True
    db_err = None
    try:
        db.session.execute(text("SELECT 1"))
    except Exception as e:
        db_ok = False
        db_err = str(e)

    # quick counts
    pred_count = None
    try:
        pred_count = db.session.query(Prediction).count()
    except Exception:
        pred_count = None

    return jsonify({
        "ok": True,
        "ts": utc_iso(now_utc()),
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
        "model_error": model_error,
        "db_ok": db_ok,
        "db_using": "postgres" if (DATABASE_URL and "postgres" in DATABASE_URL) else "sqlite",
        "db_error": db_err,
        "predictions_total": pred_count,
        "require_api_key_for_api": REQUIRE_API_KEY_FOR_API,
        "enable_auth": ENABLE_AUTH,
        "enable_rate_limit": ENABLE_RATE_LIMIT,
        "rate_limit_per_min": RATE_LIMIT_PER_MIN,
    }), 200


# ----------------------------
# Auth: request OTP
# ----------------------------
@app.route("/auth/request-code", methods=["POST"])
def request_code():
    if not ENABLE_AUTH:
        return json_error("Auth is disabled on this server", 403)

    rl = rate_limit_check()
    if rl:
        return rl

    data = request.get_json(silent=True) or {}
    email = normalize_email(data.get("email", ""))
    if not email or not EMAIL_RE.match(email):
        return json_error("Valid email is required", 400)

    # upsert user
    user = User.query.filter_by(email=email).first()
    if not user:
        user = User(email=email)
        db.session.add(user)
        db.session.commit()

    # generate OTP
    code = f"{secrets.randbelow(1_000_000):06d}"  # 6-digit
    expires_at = now_utc() + timedelta(minutes=OTP_TTL_MIN)

    # store hashed code
    rec = LoginCode(
        user_id=user.id,
        code_hash=hash_code(code),
        expires_at=expires_at,
    )
    db.session.add(rec)
    db.session.commit()

    subject = "Your Early Risk Alert AI login code"
    body = f"Your one-time login code is: {code}\n\nIt expires in {OTP_TTL_MIN} minutes."
    sent = send_email(email, subject, body)

    if sent:
        return jsonify({"ok": True, "sent": True, "expires_in_min": OTP_TTL_MIN}), 200

    # fallback: log it
    app.logger.warning(f"[OTP] email={email} code={code} expires={utc_iso(expires_at)} (SMTP not configured)")
    resp = {"ok": True, "sent": False, "expires_in_min": OTP_TTL_MIN, "note": "SMTP not configured; code logged on server"}
    if ALLOW_DEV_CODE_RETURN:
        resp["dev_code"] = code
    return jsonify(resp), 200


# ----------------------------
# Auth: verify OTP => Bearer token
# ----------------------------
@app.route("/auth/verify-code", methods=["POST"])
def verify_code():
    if not ENABLE_AUTH:
        return json_error("Auth is disabled on this server", 403)

    rl = rate_limit_check()
    if rl:
        return rl

    data = request.get_json(silent=True) or {}
    email = normalize_email(data.get("email", ""))
    code = (data.get("code", "") or "").strip()
    if not email or not EMAIL_RE.match(email):
        return json_error("Valid email is required", 400)
    if not code or len(code) < 4:
        return json_error("Valid code is required", 400)

    user = User.query.filter_by(email=email).first()
    if not user:
        return json_error("Invalid code", 401)

    # latest unused code
    rec = (LoginCode.query
           .filter_by(user_id=user.id)
           .order_by(LoginCode.id.desc())
           .first())
    if not rec:
        return json_error("Invalid code", 401)

    # already used / expired / attempts exceeded
    if rec.used_at is not None:
        return json_error("Code already used. Request a new code.", 401)
    if now_utc() > rec.expires_at:
        return json_error("Code expired. Request a new code.", 401)
    if rec.attempts >= OTP_MAX_ATTEMPTS:
        return json_error("Too many attempts. Request a new code.", 429)

    rec.attempts += 1
    db.session.commit()

    if not safe_eq(rec.code_hash, hash_code(code)):
        return json_error("Invalid code", 401)

    rec.used_at = now_utc()
    user.last_login_at = now_utc()
    db.session.commit()

    exp = time.time() + (TOKEN_TTL_DAYS * 86400)
    token = sign_token({"uid": user.id, "email": user.email, "exp": exp, "admin": bool(user.is_admin)})

    return jsonify({
        "ok": True,
        "token": token,
        "token_type": "Bearer",
        "expires_in_days": TOKEN_TTL_DAYS,
        "user": {"id": user.id, "email": user.email, "is_admin": bool(user.is_admin)}
    }), 200


# ----------------------------
# Prediction endpoint
# ----------------------------
def _predict_logic(features: np.ndarray) -> tuple[str, float]:
    """
    Returns (risk_label, probability_percent)
    """
    if model_loaded and model is not None:
        pred = int(model.predict(features)[0])
        prob_high = float(model.predict_proba(features)[0][1])  # class 1 = High Risk
        risk_label = "High Risk" if pred == 1 else "Low Risk"
        return risk_label, round(prob_high * 100.0, 2)

    # fallback heuristic if model missing (keeps API alive)
    age, bmi, ex, sys_bp, dia_bp, hr = features[0]
    score = 0.0
    if age and age >= 50: score += 1
    if bmi and bmi >= 30: score += 1
    if sys_bp and sys_bp >= 140: score += 1
    if dia_bp and dia_bp >= 90: score += 1
    if hr and hr >= 90: score += 1
    if ex is not None and ex <= 0: score += 1  # low exercise
    prob = min(95.0, max(5.0, 10.0 + score * 15.0))
    label = "High Risk" if prob >= 50 else "Low Risk"
    return label, round(prob, 2)


@app.route("/api/v1/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])  # backward compatible alias
def predict():
    rl = rate_limit_check()
    if rl:
        return rl

    ak = api_key_check()
    if ak:
        return ak

    # auth optional: if token provided, attach user; else anonymous
    _, user = auth_check(required=False)

    data = request.get_json(silent=True) or {}

    age = to_float(data.get("age"))
    bmi = to_float(data.get("bmi"))
    exercise_level = normalize_exercise_level(data.get("exercise_level"))
    systolic_bp = to_float(data.get("systolic_bp"))
    diastolic_bp = to_float(data.get("diastolic_bp"))
    heart_rate = to_float(data.get("heart_rate"))

    # basic validation
    missing = []
    for k, v in [
        ("age", age),
        ("bmi", bmi),
        ("exercise_level", exercise_level),
        ("systolic_bp", systolic_bp),
        ("diastolic_bp", diastolic_bp),
        ("heart_rate", heart_rate),
    ]:
        if v is None:
            missing.append(k)
    if missing:
        return json_error("Missing/invalid fields", 400, missing=missing)

    features = np.array([[age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate]], dtype=float)
    risk_label, probability = _predict_logic(features)

    client_id = (request.headers.get("X-Client-Id") or "").strip() or None

    # save prediction (anonymous allowed)
    try:
        rec = Prediction(
            client_id=client_id,
            user_id=user.id if user else None,
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
            risk_label=risk_label,
            probability=probability,
        )
        db.session.add(rec)
        db.session.commit()
    except Exception as e:
        app.logger.exception(f"DB save failed: {e}")

    return jsonify({
        "ok": True,
        "client_id": client_id,
        "user": {"id": user.id, "email": user.email} if user else None,
        "risk_label": risk_label,
        "probability": probability,
    }), 200


# ----------------------------
# Admin stats (optional)
# ----------------------------
def admin_basic_auth_ok() -> bool:
    if not (ADMIN_USERNAME and ADMIN_PASSWORD):
        return False
    auth = request.authorization
    if not auth:
        return False
    return safe_eq(auth.username or "", ADMIN_USERNAME) and safe_eq(auth.password or "", ADMIN_PASSWORD)

@app.route("/admin/stats", methods=["GET"])
def admin_stats():
    if not ENABLE_ADMIN:
        return json_error("Admin disabled", 403)

    # allow either admin basic auth OR admin bearer token
    u = current_user() if ENABLE_AUTH else None
    if not (admin_basic_auth_ok() or (u and u.is_admin)):
        return json_error("Unauthorized", 401)

    total = db.session.query(Prediction).count()
    authed = db.session.query(Prediction).filter(Prediction.user_id.isnot(None)).count()
    anon = total - authed

    return jsonify({
        "ok": True,
        "predictions_total": total,
        "predictions_authenticated": authed,
        "predictions_anonymous": anon,
        "rate_limit_per_min": RATE_LIMIT_PER_MIN,
        "require_api_key_for_api": REQUIRE_API_KEY_FOR_API,
        "enable_auth": ENABLE_AUTH,
    }), 200


# ============================================================
# Local run
# ============================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=env_bool("FLASK_DEBUG", False))
