import os
import datetime as dt

import numpy as np
import joblib

from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy

# ----------------------------
# App / DB setup
# ----------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev")

db_url = os.getenv("DATABASE_URL")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

# fallback local
if not db_url:
    db_url = "sqlite:///local.db"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ----------------------------
# Model load
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "demo_model.pkl")
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# ----------------------------
# DB Model
# ----------------------------
class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow, nullable=False)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.Integer, nullable=False)  # 1=Low,2=Moderate,3=High
    systolic_bp = db.Column(db.Float, nullable=False)
    diastolic_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    risk_label = db.Column(db.String(32), nullable=False)
    probability = db.Column(db.Float, nullable=False)


# ----------------------------
# Helpers
# ----------------------------
EXERCISE_MAP = {
    "Low": 1,
    "Moderate": 2,
    "Medium": 2,
    "High": 3
}

def _to_float(value, field_name):
    try:
        return float(value)
    except Exception:
        raise ValueError(f"Invalid {field_name}")

def _parse_payload(payload: dict):
    """
    Accepts either form fields or JSON keys:
    age, bmi, exercise_level, sys_bp, dia_bp, heart_rate
    """
    age = _to_float(payload.get("age"), "age")
    bmi = _to_float(payload.get("bmi"), "bmi")

    # exercise can come as string ("Low") or numeric ("1")
    exercise_raw = payload.get("exercise_level")
    if exercise_raw is None:
        raise ValueError("Missing exercise_level")

    # normalize
    if isinstance(exercise_raw, str):
        exercise_raw = exercise_raw.strip()
        if exercise_raw.isdigit():
            exercise_level = int(exercise_raw)
        else:
            exercise_level = EXERCISE_MAP.get(exercise_raw)
    else:
        exercise_level = int(exercise_raw)

    if exercise_level not in (1, 2, 3):
        raise ValueError("exercise_level must be Low/Moderate/High or 1/2/3")

    sys_bp = _to_float(payload.get("sys_bp"), "sys_bp")
    dia_bp = _to_float(payload.get("dia_bp"), "dia_bp")
    heart_rate = _to_float(payload.get("heart_rate"), "heart_rate")

    # basic sanity checks (keep it light)
    if age <= 0 or age > 120:
        raise ValueError("age out of range")
    if bmi <= 0 or bmi > 80:
        raise ValueError("bmi out of range")

    return age, bmi, exercise_level, sys_bp, dia_bp, heart_rate


def _predict(age, bmi, exercise_level, sys_bp, dia_bp, heart_rate):
    """
    If model exists: use it.
    If not: fallback simple heuristic (so app never crashes).
    """
    if model is None:
        score = 0
        score += 1 if age >= 45 else 0
        score += 1 if bmi >= 30 else 0
        score += 1 if sys_bp >= 140 else 0
        score += 1 if dia_bp >= 90 else 0
        score += 1 if heart_rate >= 90 else 0
        score += 1 if exercise_level == 1 else 0
        prob = min(0.95, 0.10 + 0.15 * score)
        pred = 1 if prob >= 0.5 else 0
        return pred, prob

    x = np.array([[age, bmi, exercise_level, sys_bp, dia_bp, heart_rate]], dtype=float)
    pred = int(model.predict(x)[0])
    prob_high = float(model.predict_proba(x)[0][1])
    return pred, prob_high


def _exercise_label(n: int) -> str:
    return {1: "Low", 2: "Moderate", 3: "High"}.get(int(n), "Unknown")


def _kpis():
    total = Prediction.query.count()
    if total == 0:
        return {"total": 0, "high_pct": 0, "avg_prob": 0}

    highs = Prediction.query.filter(Prediction.risk_label == "High Risk").count()
    avg_prob = db.session.query(db.func.avg(Prediction.probability)).scalar() or 0
    return {
        "total": int(total),
        "high_pct": round((highs / total) * 100, 1),
        "avg_prob": round(float(avg_prob) * 100, 1),
    }


# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        age, bmi, exercise_level, sys_bp, dia_bp, heart_rate = _parse_payload(request.form)
        pred, prob = _predict(age, bmi, exercise_level, sys_bp, dia_bp, heart_rate)

        risk_label = "High Risk" if pred == 1 else "Low Risk"

        row = Prediction(
            created_at=dt.datetime.utcnow(),
            age=age,
            bmi=bmi,
            exercise_level=int(exercise_level),
            systolic_bp=sys_bp,
            diastolic_bp=dia_bp,
            heart_rate=heart_rate,
            risk_label=risk_label,
            probability=float(prob),
        )
        db.session.add(row)
        db.session.commit()

        return render_template(
            "index.html",
            prediction=risk_label,
            probability=round(prob * 100, 2),
        )

    except Exception as e:
        return render_template("index.html", error=str(e)), 400


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(force=True) or {}
        age, bmi, exercise_level, sys_bp, dia_bp, heart_rate = _parse_payload(payload)
        pred, prob = _predict(age, bmi, exercise_level, sys_bp, dia_bp, heart_rate)
        risk_label = "High Risk" if pred == 1 else "Low Risk"

        row = Prediction(
            created_at=dt.datetime.utcnow(),
            age=age,
            bmi=bmi,
            exercise_level=int(exercise_level),
            systolic_bp=sys_bp,
            diastolic_bp=dia_bp,
            heart_rate=heart_rate,
            risk_label=risk_label,
            probability=float(prob),
        )
        db.session.add(row)
        db.session.commit()

        return jsonify({
            "ok": True,
            "risk_label": risk_label,
            "probability": round(prob, 6),
            "probability_percent": round(prob * 100, 2),
            "id": row.id,
            "created_at": row.created_at.isoformat() + "Z",
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/history", methods=["GET"])
def history():
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(200).all()

    kpis = _kpis()

    chart_rows = list(reversed(predictions))
    chart_labels = [p.created_at.strftime("%m/%d %H:%M") for p in chart_rows]
    chart_probs = [round(float(p.probability) * 100, 2) for p in chart_rows]
    high_count = sum(1 for p in predictions if p.risk_label == "High Risk")
    low_count = sum(1 for p in predictions if p.risk_label == "Low Risk")

    return render_template(
        "history.html",
        predictions=predictions,
        kpis=kpis,
        chart_labels=chart_labels,
        chart_probs=chart_probs,
        high_count=high_count,
        low_count=low_count,
        exercise_label=_exercise_label,
    )


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"ok": True})


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
