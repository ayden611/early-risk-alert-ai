import os
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import joblib

# ----------------------------
# App Setup
# ----------------------------

app = Flask(__name__)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev")

db_url = os.getenv("DATABASE_URL")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

if not db_url:
    db_url = "sqlite:///local.db"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ----------------------------
# Load Model
# ----------------------------

MODEL_PATH = "demo_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# ----------------------------
# Database Model
# ----------------------------

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.Integer, nullable=False)
    systolic_bp = db.Column(db.Float, nullable=False)
    diastolic_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    risk_label = db.Column(db.String(32), nullable=False)
    prob_high = db.Column(db.Float, nullable=False)

with app.app_context():
    db.create_all()

# ----------------------------
# Helpers
# ----------------------------

def normalize_exercise(level):
    mapping = {"low": 0, "moderate": 1, "high": 2}
    return mapping.get(level.lower(), 0)

def predict_risk(age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate):
    if model is None:
        return "Model Not Loaded", 0.0

    data = np.array([[age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate]])
    pred = int(model.predict(data)[0])
    prob = float(model.predict_proba(data)[0][1])

    label = "High Risk" if pred == 1 else "Low Risk"
    return label, prob

# ----------------------------
# Routes
# ----------------------------

@app.route("/health")
def health():
    return jsonify(status="ok")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            age = float(request.form.get("age", 0))
            bmi = float(request.form.get("bmi", 0))
            exercise = normalize_exercise(request.form.get("exercise_level", "low"))
            systolic = float(request.form.get("systolic_bp", 0))
            diastolic = float(request.form.get("diastolic_bp", 0))
            heart_rate = float(request.form.get("heart_rate", 0))

            label, prob = predict_risk(
                age, bmi, exercise, systolic, diastolic, heart_rate
            )

            row = Prediction(
                age=age,
                bmi=bmi,
                exercise_level=exercise,
                systolic_bp=systolic,
                diastolic_bp=diastolic,
                heart_rate=heart_rate,
                risk_label=label,
                prob_high=prob,
            )

            db.session.add(row)
            db.session.commit()

            prediction = label
            probability = round(prob * 100, 2)

        except Exception as e:
            db.session.rollback()
            error = f"{type(e).__name__}: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error,
    )

@app.route("/history")
def history():
    rows = Prediction.query.order_by(Prediction.created_at.desc()).all()
    return render_template("history.html", rows=rows)

# ----------------------------
# Local Run
# ----------------------------

if __name__ == "__main__":
    app.run(debug=True)`
