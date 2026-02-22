from flask import Blueprint, render_template, request, redirect, url_for, flash
from era.extensions import db
from era.models import Prediction
from predict import predict_risk  # <-- your root predict.py

web_bp = Blueprint("web", __name__)

def _num(form, key, label):
    v = form.get(key, "").strip()
    if not v:
        raise ValueError(f"{label} is required")
    try:
        return float(v)
    except ValueError:
        raise ValueError(f"{label} must be a number")

def _exercise(form, key="exercise_level"):
    v = form.get(key, "").strip().lower()
    if v in ("low", "0"):
        return "Low"
    if v in ("moderate", "1", "medium"):
        return "Moderate"
    if v in ("high", "2"):
        return "High"
    raise ValueError("Exercise Level must be Low, Moderate, or High")

@web_bp.get("/")
def home():
    # page load only
    return render_template("index.html", result=None, probability=None)

@web_bp.post("/predict")
def web_predict():
    """
    Server-side POST (Option B).
    This renders the result back into the HTML page.
    """
    try:
        age = _num(request.form, "age", "Age")
        bmi = _num(request.form, "bmi", "BMI")
        exercise_level = _exercise(request.form, "exercise_level")
        systolic_bp = _num(request.form, "systolic_bp", "Systolic BP")
        diastolic_bp = _num(request.form, "diastolic_bp", "Diastolic BP")
        heart_rate = _num(request.form, "heart_rate", "Heart Rate")

        risk_label, probability = predict_risk(
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
        )

        # Save to DB (same as API)
        pred = Prediction(
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
            risk_label=risk_label,
            probability=probability,
        )
        db.session.add(pred)
        db.session.commit()

        return render_template("index.html", result=risk_label, probability=probability)

    except Exception as e:
        flash(str(e))
        return redirect(url_for("web.home"))

@web_bp.get("/history")
def history():
    rows = Prediction.query.order_by(Prediction.created_at.desc()).limit(50).all()
    return render_template("history.html", rows=rows)
