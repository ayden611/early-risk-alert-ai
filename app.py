import os
import joblib
from datetime import datetime, timezone

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy

# ----------------------------
# App + DB setup
# ----------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")  # change in prod

DATABASE_URL = os.getenv("DATABASE_URL")  # Render usually provides this
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL or "sqlite:///local.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ----------------------------
# Model (adjust filename if needed)
# ----------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

# Exercise mapping (store numeric in DB)
EXERCISE_MAP = {"Low": 1, "Medium": 2, "High": 3}
EXERCISE_REVERSE = {v: k for k, v in EXERCISE_MAP.items()}


# ----------------------------
# DB Model
# ----------------------------
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.Integer, nullable=False)

    systolic_bp = db.Column(db.Float, nullable=False)
    diastolic_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    risk_label = db.Column(db.String(50), nullable=False)
    probability = db.Column(db.Float, nullable=False)


with app.app_context():
    db.create_all()


# ----------------------------
# Helpers
# ----------------------------
def to_float(name: str, min_val=None, max_val=None):
    raw = request.form.get(name, "").strip()
    if raw == "":
        raise ValueError(f"{name} is required")
    try:
        val = float(raw)
    except Exception:
        raise ValueError(f"{name} must be a number")

    if min_val is not None and val < min_val:
        raise ValueError(f"{name} must be ≥ {min_val}")
    if max_val is not None and val > max_val:
        raise ValueError(f"{name} must be ≤ {max_val}")
    return val


def run_model(age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate):
    """
    Returns: (risk_label, probability)
    probability = probability of high-risk (or positive class)
    """
    if model is None:
        score = 0
        if age >= 45: score += 1
        if bmi >= 30: score += 1
        if systolic_bp >= 140: score += 1
        if diastolic_bp >= 90: score += 1
        if heart_rate >= 100: score += 1
        if exercise_level == 1: score += 1

        prob = min(0.95, max(0.05, score / 6))
        label = "High Risk" if prob >= 0.5 else "Low Risk"
        return label, float(prob)

    X = [[age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate]]

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])
    else:
        pred = int(model.predict(X)[0])
        proba = 0.9 if pred == 1 else 0.1

    label = "High Risk" if proba >= 0.5 else "Low Risk"
    return label, proba


# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = to_float("age", 1, 120)
        bmi = to_float("bmi", 10, 80)
        systolic_bp = to_float("systolic_bp", 70, 250)
        diastolic_bp = to_float("diastolic_bp", 40, 160)
        heart_rate = to_float("heart_rate", 30, 220)

        exercise_str = request.form.get("exercise_level", "Low")
        if exercise_str not in EXERCISE_MAP:
            raise ValueError("exercise_level must be Low, Medium, or High")
        exercise_level = EXERCISE_MAP[exercise_str]

        risk_label, probability = run_model(
            age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate
        )

        p = Prediction(
            created_at=datetime.now(timezone.utc),
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
            risk_label=risk_label,
            probability=probability,
        )
        db.session.add(p)
        db.session.commit()

        return render_template(
            "index.html",
            result=risk_label,
            probability=f"{probability*100:.2f}%",
            form_data={
                "age": age,
                "bmi": bmi,
                "exercise_level": exercise_str,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "heart_rate": heart_rate,
            },
        )

    except Exception as e:
        db.session.rollback()
        flash(str(e), "error")
        return redirect(url_for("home"))


@app.route("/history", methods=["GET"])
def history():
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(50).all()
    rows = []
    for p in predictions:
        rows.append(
            {
                "created_at": p.created_at,
                "age": p.age,
                "bmi": p.bmi,
                "exercise": EXERCISE_REVERSE.get(p.exercise_level, str(p.exercise_level)),
                "risk_label": p.risk_label,
                "probability": p.probability,
            }
        )
    return render_template("history.html", predictions=rows)


if __name__ == "__main__":
    app.run(debug=True)
