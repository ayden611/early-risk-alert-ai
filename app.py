import os
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy

# Your predictor (you already have predict.py)
from predict import predict_risk


# ----------------------------
# App + Config
# ----------------------------
app = Flask(__name__)

# Secret key (needed if you ever use sessions/flash; harmless to have)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")

# Render provides DATABASE_URL. Locally we fallback to sqlite.
db_url = (os.environ.get("DATABASE_URL") or "").strip()
if db_url.startswith("postgres://"):
    # SQLAlchemy expects postgresql://
    db_url = db_url.replace("postgres://", "postgresql://", 1)

if not db_url:
    db_url = "sqlite:///local.db"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# ----------------------------
# DB Model
# ----------------------------
class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.String(32), nullable=False)

    systolic_bp = db.Column(db.Float, nullable=False)
    diastolic_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    risk_label = db.Column(db.String(32), nullable=False)
    prob_high = db.Column(db.Float, nullable=False)  # 0..1


with app.app_context():
    db.create_all()


# ----------------------------
# Helpers
# ----------------------------
def normalize_exercise(val: str) -> str:
    v = (val or "").strip().lower()
    if v in ("low", "l"):
        return "Low"
    if v in ("moderate", "mod", "m", "medium"):
        return "Moderate"
    if v in ("high", "h"):
        return "High"
    # fallback
    return "Low"


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/debug")
def debug():
    """
    Proves what DB you're connected to + how many rows exist.
    """
    try:
        count = db.session.query(Prediction).count()
    except Exception as e:
        return jsonify(error=str(e)), 500

    # Don’t leak full credentials; show a safe version
    safe_db = app.config["SQLALCHEMY_DATABASE_URI"]
    if "@" in safe_db and "://" in safe_db:
        # redact password between :// and @ crudely
        # e.g. postgresql://user:pass@host/db -> postgresql://user:***@host/db
        prefix, rest = safe_db.split("://", 1)
        if "@" in rest and ":" in rest.split("@")[0]:
            user_part, host_part = rest.split("@", 1)
            user = user_part.split(":", 1)[0]
            safe_db = f"{prefix}://{user}:***@{host_part}"

    return jsonify(db_uri=safe_db, count=count)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            age = float(request.form.get("age", "0") or "0")
            bmi = float(request.form.get("bmi", "0") or "0")
            exercise_level = normalize_exercise(request.form.get("exercise_level", "Low"))

            systolic_bp = float(request.form.get("systolic_bp", "0") or "0")
            diastolic_bp = float(request.form.get("diastolic_bp", "0") or "0")
            heart_rate = float(request.form.get("heart_rate", "0") or "0")

            label, prob_high = predict_risk(
                age=age,
                bmi=bmi,
                exercise_level=exercise_level,
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                heart_rate=heart_rate,
            )

            # Save to DB
            row = Prediction(
                age=age,
                bmi=bmi,
                exercise_level=exercise_level,
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                heart_rate=heart_rate,
                risk_label=str(label),
                prob_high=float(prob_high),
            )
            db.session.add(row)
            db.session.commit()

            prediction = str(label)
            probability = round(float(prob_high) * 100.0, 2)

        except Exception as e:
            db.session.rollback()
            error = f"{type(e).__name__}: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error,
    )


@app.get("/history")
def history():
    rows = (
        Prediction.query.order_by(Prediction.created_at.desc())
        .limit(200)
        .all()
    )
    # IMPORTANT: history.html in your repo used "records" earlier, not "rows"
    return render_template("history.html", records=rows)


# Instant(JS) endpoint (optional, but keeps your UI working if you use it)
@app.post("/api/predict")
def api_predict():
    try:
        data = request.get_json(force=True) or {}

        age = float(data.get("age", 0) or 0)
        bmi = float(data.get("bmi", 0) or 0)
        exercise_level = normalize_exercise(data.get("exercise_level", "Low"))

        systolic_bp = float(data.get("systolic_bp", 0) or 0)
        diastolic_bp = float(data.get("diastolic_bp", 0) or 0)
        heart_rate = float(data.get("heart_rate", 0) or 0)

        label, prob_high = predict_risk(
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
        )

        # Save to DB
        row = Prediction(
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
            risk_label=str(label),
            prob_high=float(prob_high),
        )
        db.session.add(row)
        db.session.commit()

        return jsonify(
            risk=str(label),
            prob_high=float(prob_high),
            probability=round(float(prob_high) * 100.0, 2),
        )

    except Exception as e:
        db.session.rollback()
        return jsonify(error=f"{type(e).__name__}: {e}"), 500`
