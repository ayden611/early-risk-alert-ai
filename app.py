import os
import csv
import io
import json
import joblib
import numpy as np
import datetime as dt
import time
import logging
from sqlachemy import text

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    Response,
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user,
)

# ----------------------------
# App + DB setup
# ----------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL or "sqlite:///local.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
# ============================
# Production foundation
# ============================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger("early-risk-alert-ai")

START_TIME = time.time()

MODEL_META_PATH = os.getenv("MODEL_META_PATH", "model_meta.json")


def _load_model_meta():
    try:
        with open(MODEL_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


logger.info("App booting")
logger.info("LOG_LEVEL=%s", LOG_LEVEL)
logger.info("DATABASE_URL set? %s", bool(os.getenv("DATABASE_URL")))

# ----------------------------
# Auth (Flask-Login)
# ----------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "change-me")  # set this in Render env vars


class User(UserMixin):
    def __init__(self, user_id: str):
        self.id = user_id


@login_manager.user_loader
def load_user(user_id):
    if user_id == ADMIN_USERNAME:
        return User(user_id)
    return None


# ----------------------------
# Model load
# ----------------------------
MODEL_PATH = "demo_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# ----------------------------
# Helpers
# ----------------------------
EXERCISE_MAP = {"Low": 1, "Medium": 2, "High": 3}


def safe_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v, default=None):
    try:
        return int(float(v))
    except Exception:
        return default


def exercise_to_numeric(v):
    """
    Accepts:
      - "Low"/"Medium"/"High"
      - "1"/"2"/"3"
      - 1/2/3
    Returns int 1..3 or None
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        iv = int(v)
        return iv if iv in (1, 2, 3) else None
    s = str(v).strip()
    if s in EXERCISE_MAP:
        return EXERCISE_MAP[s]
    iv = safe_int(s)
    return iv if iv in (1, 2, 3) else None


def exercise_to_label(v):
    inv = {1: "Low", 2: "Medium", 3: "High"}
    try:
        iv = int(v)
        return inv.get(iv, "Unknown")
    except Exception:
        return "Unknown"


def build_features(age, bmi, exercise_level, sys_bp, dia_bp, hr):
    return np.array([[age, bmi, exercise_level, sys_bp, dia_bp, hr]], dtype=float)


def predict_internal(age, bmi, exercise_level, sys_bp, dia_bp, hr):
    if model is None:
        raise RuntimeError("Model file demo_model.pkl not found on server.")
    X = build_features(age, bmi, exercise_level, sys_bp, dia_bp, hr)
    pred = int(model.predict(X)[0])
    prob_high = float(model.predict_proba(X)[0][1])
    risk_label = "High Risk" if pred == 1 else "Low Risk"
    return risk_label, prob_high


# ----------------------------
# DB Model
# ----------------------------
class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow, nullable=False)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.Integer, nullable=False)  # 1..3
    systolic_bp = db.Column(db.Float, nullable=False)
    diastolic_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    risk_label = db.Column(db.String(50), nullable=False)
    probability = db.Column(db.Float, nullable=False)  # probability of High Risk 0..1


with app.app_context():
    db.create_all()


# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability_pct = None

    if request.method == "POST":
        age = safe_float(request.form.get("age"))
        bmi = safe_float(request.form.get("bmi"))
        exercise_level = exercise_to_numeric(request.form.get("exercise_level"))
        sys_bp = safe_float(request.form.get("systolic_bp"))
        dia_bp = safe_float(request.form.get("diastolic_bp"))
        heart_rate = safe_float(request.form.get("heart_rate"))

        missing = [
            k
            for k, v in {
                "age": age,
                "bmi": bmi,
                "exercise_level": exercise_level,
                "systolic_bp": sys_bp,
                "diastolic_bp": dia_bp,
                "heart_rate": heart_rate,
            }.items()
            if v is None
        ]

        if missing:
            flash(f"Missing/invalid: {', '.join(missing)}", "error")
            return redirect(url_for("home"))

        risk_label, prob_high = predict_internal(
            age, bmi, exercise_level, sys_bp, dia_bp, heart_rate
        )

        # Save to DB
        p = Prediction(
            created_at=dt.datetime.utcnow(),
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=sys_bp,
            diastolic_bp=dia_bp,
            heart_rate=heart_rate,
            risk_label=risk_label,
            probability=prob_high,
        )
        db.session.add(p)
        db.session.commit()

        prediction = risk_label
        probability_pct = round(prob_high * 100, 2)

    return render_template(
        "index.html", prediction=prediction, probability=probability_pct
    )


@app.route("/history")
@login_required
def history():
    # Pagination
    page = safe_int(request.args.get("page", 1), 1)
    per_page = safe_int(request.args.get("per_page", 10), 10)
    per_page = max(5, min(per_page, 50))

    q = Prediction.query.order_by(Prediction.created_at.desc())
    total = q.count()
    pages = max(1, (total + per_page - 1) // per_page)
    page = max(1, min(page, pages))

    rows = q.offset((page - 1) * per_page).limit(per_page).all()

    # KPIs
    if total > 0:
        high_count = Prediction.query.filter(
            Prediction.risk_label == "High Risk"
        ).count()
        high_pct = (high_count / total) * 100.0
        avg_prob = db.session.query(db.func.avg(Prediction.probability)).scalar() or 0.0
    else:
        high_count = 0
        high_pct = 0.0
        avg_prob = 0.0

    since = dt.datetime.utcnow() - dt.timedelta(hours=24)
    last_24h = Prediction.query.filter(Prediction.created_at >= since).count()

    # Charts
    # 1) Risk split
    low_count = Prediction.query.filter(Prediction.risk_label == "Low Risk").count()
    risk_split = {"Low Risk": low_count, "High Risk": high_count}

    # 2) Probability trend (last 30)
    last30 = Prediction.query.order_by(Prediction.created_at.desc()).limit(30).all()
    last30 = list(reversed(last30))
    trend_labels = [p.created_at.strftime("%m-%d %H:%M") for p in last30]
    trend_probs = [round(p.probability * 100.0, 2) for p in last30]

    return render_template(
        "history.html",
        rows=rows,
        total=total,
        high_pct=round(high_pct, 1),
        avg_prob=round(avg_prob * 100.0, 1),
        last_24h=last_24h,
        page=page,
        pages=pages,
        per_page=per_page,
        risk_split_json=json.dumps(risk_split),
        trend_labels_json=json.dumps(trend_labels),
        trend_probs_json=json.dumps(trend_probs),
        exercise_to_label=exercise_to_label,
    )


@app.route("/export.csv")
@login_required
def export_csv():
    q = Prediction.query.order_by(Prediction.created_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "created_at",
            "age",
            "bmi",
            "exercise_level",
            "systolic_bp",
            "diastolic_bp",
            "heart_rate",
            "risk_label",
            "probability_high_risk",
        ]
    )
    for p in q:
        writer.writerow(
            [
                p.created_at.isoformat(),
                p.age,
                p.bmi,
                exercise_to_label(p.exercise_level),
                p.systolic_bp,
                p.diastolic_bp,
                p.heart_rate,
                p.risk_label,
                round(p.probability, 6),
            ]
        )

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API:
    POST /api/predict
    {
      "age": 50,
      "bmi": 25.0,
      "exercise_level": "Low" or 1,
      "systolic_bp": 120,
      "diastolic_bp": 80,
      "heart_rate": 70
    }
    Returns:
    {
      "risk_label": "High Risk"|"Low Risk",
      "probability_high_risk": 0.1234,
      "probability_high_risk_pct": 12.34
    }
    """
    data = request.get_json(silent=True) or {}

    age = safe_float(data.get("age"))
    bmi = safe_float(data.get("bmi"))
    exercise_level = exercise_to_numeric(data.get("exercise_level"))
    sys_bp = safe_float(data.get("systolic_bp"))
    dia_bp = safe_float(data.get("diastolic_bp"))
    hr = safe_float(data.get("heart_rate"))

    missing = [
        k
        for k, v in {
            "age": age,
            "bmi": bmi,
            "exercise_level": exercise_level,
            "systolic_bp": sys_bp,
            "diastolic_bp": dia_bp,
            "heart_rate": hr,
        }.items()
        if v is None
    ]

    if missing:
        return jsonify({"error": "Missing/invalid fields", "fields": missing}), 400

    try:
        risk_label, prob_high = predict_internal(
            age, bmi, exercise_level, sys_bp, dia_bp, hr
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Optional: save API predictions too (comment out if you don't want that)
    p = Prediction(
        created_at=dt.datetime.utcnow(),
        age=age,
        bmi=bmi,
        exercise_level=exercise_level,
        systolic_bp=sys_bp,
        diastolic_bp=dia_bp,
        heart_rate=hr,
        risk_label=risk_label,
        probability=prob_high,
    )
    db.session.add(p)
    db.session.commit()

    return jsonify(
        {
            "risk_label": risk_label,
            "probability_high_risk": round(prob_high, 6),
            "probability_high_risk_pct": round(prob_high * 100.0, 2),
        }
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("history"))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            login_user(User(ADMIN_USERNAME))
            return redirect(url_for("history"))

        flash("Invalid login.", "error")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out.", "ok")
    return redirect(url_for("home"))


# Health check
@app.route("/health")
def health():
    return {"ok": True}


@app.get("/healthz")
def healthz():
    db_ok = True
    db_error = None

    try:
        db.session.execute(text("SELECT 1"))
    except Exception as e:
        db_ok = False
        db_error = str(e)

    meta = _load_model_meta()

    return {
        "status": "ok" if db_ok else "degraded",
        "db_ok": db_ok,
        "db_error": db_error,
        "model_loaded": model is not None,
        "model_meta": {
            "version": meta.get("version"),
            "trained_at": meta.get("trained_at"),
            "features": meta.get("features"),
        },
        "uptime_seconds": round(time.time() - START_TIME, 2),
    }, (200 if db_ok else 503)
