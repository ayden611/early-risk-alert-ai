import os
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import numpy as np
import joblib

# --------------------------------------------------
# App Setup
# --------------------------------------------------

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev")

    db_url = os.getenv("DATABASE_URL")

    # Render uses postgres:// sometimes; SQLAlchemy needs postgresql://
    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    if not db_url:
        db_url = "sqlite:///local.db"

    app.config["SQLALCHEMY_DATABASE_URI"] = db_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    return app

app = create_app()
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# --------------------------------------------------
# DB Model
# --------------------------------------------------

class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.Float, nullable=False)

    systolic_bp = db.Column(db.Float, nullable=False)
    diastolic_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    risk_label = db.Column(db.String(20), nullable=False)
    probability = db.Column(db.Float, nullable=False)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

MODEL_PATH = "demo_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

FEATURES = ["age", "bmi", "exercise_level", "systolic_bp", "diastolic_bp", "heart_rate"]

def _to_float(val, default=None):
    try:
        if val is None:
            return default
        if isinstance(val, str) and val.strip() == "":
            return default
        return float(val)
    except Exception:
        return default

def predict_one(payload: dict):
    """
    Returns (label, prob_high)
    """
    if model is None:
        return ("Low Risk", 0.0)

    x = np.array([[
        _to_float(payload.get("age"), 0),
        _to_float(payload.get("bmi"), 0),
        _to_float(payload.get("exercise_level"), 0),
        _to_float(payload.get("systolic_bp"), 0),
        _to_float(payload.get("diastolic_bp"), 0),
        _to_float(payload.get("heart_rate"), 0),
    ]], dtype=float)

    pred = int(model.predict(x)[0])
    prob_high = float(model.predict_proba(x)[0][1]) if hasattr(model, "predict_proba") else float(pred)

    label = "High Risk" if pred == 1 else "Low Risk"
    return label, prob_high

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    error = None

    if request.method == "POST":
        payload = {
            "age": request.form.get("age"),
            "bmi": request.form.get("bmi"),
            "exercise_level": request.form.get("exercise_level"),
            "systolic_bp": request.form.get("systolic_bp"),
            "diastolic_bp": request.form.get("diastolic_bp"),
            "heart_rate": request.form.get("heart_rate"),
        }

        try:
            label, prob_high = predict_one(payload)

            row = Prediction(
                age=_to_float(payload["age"], 0),
                bmi=_to_float(payload["bmi"], 0),
                exercise_level=_to_float(payload["exercise_level"], 0),
                systolic_bp=_to_float(payload["systolic_bp"], 0),
                diastolic_bp=_to_float(payload["diastolic_bp"], 0),
                heart_rate=_to_float(payload["heart_rate"], 0),
                risk_label=label,
                probability=prob_high,
            )
            db.session.add(row)
            db.session.commit()

            result = label
            probability = round(prob_high * 100, 2)

        except Exception as e:
            db.session.rollback()
            error = str(e)

    return render_template("index.html", result=result, probability=probability, error=error)

@app.route("/history", methods=["GET"])
def history():
    rows = Prediction.query.order_by(Prediction.created_at.desc()).limit(50).all()
    return render_template("history.html", rows=rows)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

# --------------------------------------------------
# Auto-create tables
# --------------------------------------------------

@app.before_request
def ensure_tables():
    try:
        db.create_all()
    except Exception:
        pass
