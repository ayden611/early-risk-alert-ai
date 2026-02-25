import os
import json
import secrets
import hashlib
from datetime import datetime, timezone

import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template_string, abort
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

# =========================
# App + Config
# =========================
app = Flask(__name__)

# --- Secrets / env ---
app.secret_key = os.getenv("SECRET_KEY", "dev")

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL or "sqlite:///local.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

MODEL_PATH = os.getenv("MODEL_PATH", "demo_model.pkl")

# Gate API with X-API-Key header? (for consumer apps, you can leave false)
REQUIRE_API_KEY_FOR_API = (
    os.getenv("REQUIRE_API_KEY_FOR_API", "false").strip().lower() in ("1", "true", "yes", "y")
)

# Optional static allowlist of keys in env (comma-separated).
# Example: API_KEYS=era_xxx,era_yyy
API_KEYS_ENV = [k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()]

# Admin credentials (only used for /keys/* endpoints)
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")

db = SQLAlchemy(app)

# =========================
# DB Models
# =========================
class APIKey(db.Model):
    __tablename__ = "api_key"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    key_hash = db.Column(db.String(128), nullable=False, unique=True, index=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

class Prediction(db.Model):
    __tablename__ = "prediction"
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # anonymous tracking
    client_id = db.Column(db.String(120), nullable=True)
    user_id = db.Column(db.String(120), nullable=True)

    # inputs
    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.String(32), nullable=False)   # TEXT so it won't crash on "Moderate"
    systolic_bp = db.Column(db.Float, nullable=False)
    diastolic_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    # outputs
    risk_label = db.Column(db.String(32), nullable=False)
    probability = db.Column(db.Float, nullable=False)

# =========================
# Model Load
# =========================
model = None
model_error = None
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        model_error = f"MODEL_PATH not found: {MODEL_PATH}"
except Exception as e:
    model_error = f"Failed to load model: {e}"

# =========================
# Helpers
# =========================
def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def normalize_exercise_level(v):
    """
    Accepts: "Low/Moderate/High", "0/1/2", numbers, etc.
    Returns: (label_str, numeric_float_for_model)
    """
    if v is None:
        return ("Moderate", 1.0)

    # numeric?
    try:
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.strip().replace(".", "", 1).isdigit()):
            n = float(v)
            if n <= 0:
                return ("Low", 0.0)
            if n >= 2:
                return ("High", 2.0)
            # 1-ish
            return ("Moderate", 1.0)
    except Exception:
        pass

    s = str(v).strip().lower()
    mapping = {
        "low": ("Low", 0.0),
        "l": ("Low", 0.0),
        "0": ("Low", 0.0),
        "sedentary": ("Low", 0.0),

        "moderate": ("Moderate", 1.0),
        "mod": ("Moderate", 1.0),
        "m": ("Moderate", 1.0),
        "1": ("Moderate", 1.0),
        "medium": ("Moderate", 1.0),

        "high": ("High", 2.0),
        "h": ("High", 2.0),
        "2": ("High", 2.0),
        "active": ("High", 2.0),
    }
    return mapping.get(s, ("Moderate", 1.0))

def get_request_api_key() -> str | None:
    # Accept either header
    return request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer", "").strip() or None

def api_key_is_valid(raw_key: str) -> bool:
    if not raw_key:
        return False

    # 1) env allowlist (direct compare)
    if API_KEYS_ENV and raw_key in API_KEYS_ENV:
        return True

    # 2) DB allowlist (hash compare)
    key_hash = sha256_hex(raw_key)
    row = APIKey.query.filter_by(key_hash=key_hash, is_active=True).first()
    return row is not None

def require_api_key_if_enabled():
    if not REQUIRE_API_KEY_FOR_API:
        return
    raw = get_request_api_key()
    if not raw or not api_key_is_valid(raw):
        return jsonify({"error": "unauthorized", "message": "Valid X-API-Key required"}), 401

def require_admin_basic_auth():
    # If admin creds not set, lock down key mgmt completely
    if not ADMIN_USERNAME or not ADMIN_PASSWORD:
        return jsonify({"error": "admin_not_configured", "message": "Set ADMIN_USERNAME and ADMIN_PASSWORD"}), 403

    auth = request.authorization
    if not auth or auth.username != ADMIN_USERNAME or auth.password != ADMIN_PASSWORD:
        resp = jsonify({"error": "auth_required", "message": "Basic auth required"})
        resp.status_code = 401
        resp.headers["WWW-Authenticate"] = 'Basic realm="Admin"'
        return resp
    return None

def safe_float(x, field_name: str):
    try:
        return float(x)
    except Exception:
        raise ValueError(f"Invalid {field_name}: must be a number")

# =========================
# DB Init (creates tables if missing)
# =========================
with app.app_context():
    db.create_all()

# =========================
# Routes
# =========================
@app.route("/healthz", methods=["GET"])
def healthz():
    db_ok = True
    db_error = None
    try:
        db.session.execute(text("SELECT 1"))
    except Exception as e:
        db_ok = False
        db_error = str(e)

    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_error": model_error,
        "db_ok": db_ok,
        "db_error": db_error,
        "require_api_key_for_api": REQUIRE_API_KEY_FOR_API,
        "uptime_seconds": 0
    })

@app.route("/", methods=["GET", "POST"])
def home():
    # Simple UI without templates (one-file app)
    result = None
    err = None

    if request.method == "POST":
        try:
            age = safe_float(request.form.get("age"), "age")
            bmi = safe_float(request.form.get("bmi"), "bmi")
            ex_label, ex_num = normalize_exercise_level(request.form.get("exercise_level"))
            sys_bp = safe_float(request.form.get("systolic_bp"), "systolic_bp")
            dia_bp = safe_float(request.form.get("diastolic_bp"), "diastolic_bp")
            hr = safe_float(request.form.get("heart_rate"), "heart_rate")

            if model is None:
                raise RuntimeError(model_error or "Model not loaded")

            X = np.array([[age, bmi, ex_num, sys_bp, dia_bp, hr]], dtype=float)
            pred = int(model.predict(X)[0])
            prob_high = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)

            risk_label = "High Risk" if pred == 1 else "Low Risk"
            probability = round(prob_high * 100.0, 2)

            # anonymous save
            client_id = request.headers.get("X-Client-Id") or request.form.get("client_id") or None
            row = Prediction(
                client_id=client_id,
                user_id=None,
                age=age,
                bmi=bmi,
                exercise_level=ex_label,
                systolic_bp=sys_bp,
                diastolic_bp=dia_bp,
                heart_rate=hr,
                risk_label=risk_label,
                probability=probability
            )
            db.session.add(row)
            db.session.commit()

            result = {"risk_label": risk_label, "probability": probability, "exercise_level": ex_label}
        except Exception as e:
            err = str(e)

    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>Early Risk Alert AI</title>
      <style>
        body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:0;background:#0b1220;color:#e9eefc}
        .wrap{max-width:920px;margin:32px auto;padding:0 16px}
        .card{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:16px}
        label{display:block;font-size:13px;color:#8ea0c6;margin:10px 0 6px}
        input,select{width:100%;padding:10px;border-radius:12px;border:1px solid rgba(255,255,255,.12);background:#121b2e;color:#e9eefc}
        .grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
        .btn{margin-top:14px;padding:12px 14px;border-radius:12px;border:0;background:#2b6cff;color:white;font-weight:700;cursor:pointer}
        .row{display:flex;gap:10px;align-items:center;justify-content:space-between}
        .muted{color:#8ea0c6}
        .pill{padding:6px 10px;border-radius:999px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.08)}
        a{color:#9bb7ff;text-decoration:none}
        .err{color:#ffb4b4}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="row">
          <h2 style="margin:0">Early Risk Alert AI</h2>
          <div class="row">
            <a class="pill" href="/history">History</a>
            <a class="pill" href="/healthz">Health</a>
          </div>
        </div>

        <div class="card" style="margin-top:14px">
          <form method="post">
            <div class="grid">
              <div>
                <label>Age</label>
                <input name="age" placeholder="45" required>
              </div>
              <div>
                <label>BMI</label>
                <input name="bmi" placeholder="27.5" required>
              </div>
              <div>
                <label>Exercise Level</label>
                <select name="exercise_level">
                  <option>Low</option>
                  <option selected>Moderate</option>
                  <option>High</option>
                </select>
              </div>
              <div>
                <label>Systolic BP</label>
                <input name="systolic_bp" placeholder="130" required>
              </div>
              <div>
                <label>Diastolic BP</label>
                <input name="diastolic_bp" placeholder="85" required>
              </div>
              <div>
                <label>Heart Rate</label>
                <input name="heart_rate" placeholder="72" required>
              </div>
            </div>

            <label class="muted">Client ID (optional, for anonymous tracking)</label>
            <input name="client_id" placeholder="test-client-1">

            <button class="btn" type="submit">Predict Risk</button>
          </form>

          {% if err %}
            <p class="err"><b>Error:</b> {{err}}</p>
          {% endif %}

          {% if result %}
            <div class="card" style="margin-top:14px">
              <div class="row">
                <div>
                  <div class="muted">Risk Label</div>
                  <div style="font-size:22px;font-weight:800">{{result.risk_label}}</div>
                </div>
                <div>
                  <div class="muted">Probability (High Risk)</div>
                  <div style="font-size:22px;font-weight:800">{{result.probability}}%</div>
                </div>
                <div>
                  <div class="muted">Exercise</div>
                  <div style="font-size:22px;font-weight:800">{{result.exercise_level}}</div>
                </div>
              </div>
            </div>
          {% endif %}
        </div>

        <p class="muted" style="margin-top:14px">
          API endpoint: <span class="pill">POST /api/predict</span>
          {% if require_key %}<span class="pill">X-API-Key required</span>{% else %}<span class="pill">anonymous allowed</span>{% endif %}
        </p>
      </div>
    </body>
    </html>
    """
    return render_template_string(html, result=result, err=err, require_key=REQUIRE_API_KEY_FOR_API)

@app.route("/history", methods=["GET"])
def history():
    rows = Prediction.query.order_by(Prediction.id.desc()).limit(50).all()
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <title>History - Early Risk Alert AI</title>
      <style>
        body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:0;background:#0b1220;color:#e9eefc}
        .wrap{max-width:920px;margin:32px auto;padding:0 16px}
        .card{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:16px}
        table{width:100%;border-collapse:collapse}
        th,td{padding:10px;border-bottom:1px solid rgba(255,255,255,.08);font-size:14px}
        th{text-align:left;color:#8ea0c6;font-weight:600}
        a{color:#9bb7ff;text-decoration:none}
        .pill{padding:6px 10px;border-radius:999px;background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.08)}
        .row{display:flex;gap:10px;align-items:center;justify-content:space-between}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="row">
          <h2 style="margin:0">Prediction History</h2>
          <a class="pill" href="/">Back</a>
        </div>
        <div class="card" style="margin-top:14px">
          <table>
            <thead>
              <tr>
                <th>Time (UTC)</th>
                <th>Client</th>
                <th>Risk</th>
                <th>Prob%</th>
                <th>Age</th>
                <th>BMI</th>
                <th>Ex</th>
                <th>BP</th>
                <th>HR</th>
              </tr>
            </thead>
            <tbody>
              {% for r in rows %}
              <tr>
                <td>{{r.created_at}}</td>
                <td>{{r.client_id or "-"}}</td>
                <td>{{r.risk_label}}</td>
                <td>{{"%.2f"|format(r.probability)}}</td>
                <td>{{r.age}}</td>
                <td>{{r.bmi}}</td>
                <td>{{r.exercise_level}}</td>
                <td>{{r.systolic_bp}}/{{r.diastolic_bp}}</td>
                <td>{{r.heart_rate}}</td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
    </body>
    </html>
    """
    return render_template_string(html, rows=rows)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    gate = require_api_key_if_enabled()
    if gate is not None:
        return gate

    if model is None:
        return jsonify({"error": "model_not_loaded", "message": model_error or "Model not loaded"}), 500

    payload = request.get_json(silent=True) or {}
    try:
        age = safe_float(payload.get("age"), "age")
        bmi = safe_float(payload.get("bmi"), "bmi")
        ex_label, ex_num = normalize_exercise_level(payload.get("exercise_level"))
        sys_bp = safe_float(payload.get("systolic_bp"), "systolic_bp")
        dia_bp = safe_float(payload.get("diastolic_bp"), "diastolic_bp")
        hr = safe_float(payload.get("heart_rate"), "heart_rate")

        X = np.array([[age, bmi, ex_num, sys_bp, dia_bp, hr]], dtype=float)
        pred = int(model.predict(X)[0])
        prob_high = float(model.predict_proba(X)[0][1]) if hasattr(model, "predict_proba") else (1.0 if pred == 1 else 0.0)

        risk_label = "High Risk" if pred == 1 else "Low Risk"
        probability = round(prob_high * 100.0, 2)

        client_id = request.headers.get("X-Client-Id") or payload.get("client_id") or None

        row = Prediction(
            client_id=client_id,
            user_id=None,
            age=age,
            bmi=bmi,
            exercise_level=ex_label,
            systolic_bp=sys_bp,
            diastolic_bp=dia_bp,
            heart_rate=hr,
            risk_label=risk_label,
            probability=probability
        )
        db.session.add(row)
        db.session.commit()

        return jsonify({
            "client_id": client_id,
            "probability": probability,
            "risk_label": risk_label
        })
    except Exception as e:
        return jsonify({"error": "bad_request", "message": str(e)}), 400

@app.route("/keys/create", methods=["POST"])
def keys_create():
    # Admin-only endpoint to create a DB-backed API key.
    # Use Basic Auth: -u ADMIN_USERNAME:ADMIN_PASSWORD
    auth_resp = require_admin_basic_auth()
    if auth_resp is not None:
        return auth_resp

    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "default").strip()

    raw_key = "era_" + secrets.token_hex(32)
    key_hash = sha256_hex(raw_key)

    # store hash only
    row = APIKey(name=name, key_hash=key_hash, is_active=True)
    db.session.add(row)
    db.session.commit()

    return jsonify({
        "name": name,
        "api_key": raw_key,     # returned ONCE; store it safely
        "created_at": row.created_at.isoformat()
    }), 201

@app.route("/keys/list", methods=["GET"])
def keys_list():
    auth_resp = require_admin_basic_auth()
    if auth_resp is not None:
        return auth_resp

    rows = APIKey.query.order_by(APIKey.id.desc()).limit(50).all()
    return jsonify([{
        "id": r.id,
        "name": r.name,
        "is_active": r.is_active,
        "created_at": r.created_at.isoformat()
    } for r in rows])

@app.route("/keys/deactivate/<int:key_id>", methods=["POST"])
def keys_deactivate(key_id: int):
    auth_resp = require_admin_basic_auth()
    if auth_resp is not None:
        return auth_resp

    row = APIKey.query.get(key_id)
    if not row:
        return jsonify({"error": "not_found"}), 404
    row.is_active = False
    db.session.commit()
    return jsonify({"ok": True, "id": key_id})

# =========================
# Entry
# =========================
if __name__ == "__main__":
    # Local dev only
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")), debug=True)
