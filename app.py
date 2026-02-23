import os
from datetime import datetime

import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy

# -------------------------------------------------
# App Setup
# -------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------
# Database Setup (Render Postgres + Local fallback)
# -------------------------------------------------
db_url = os.getenv("DATABASE_URL", "").strip()

# Fix Render's postgres:// issue
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

# Local fallback (only used if no DATABASE_URL exists)
if not db_url:
    db_url = "sqlite:///local.db"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# -------------------------------------------------
# Load ML Model
# -------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "demo_model.pkl")
model = joblib.load(MODEL_PATH)

EXERCISE_MAP = {
    "low": 0,
    "moderate": 1,
    "high": 2,
    "0": 0,
    "1": 1,
    "2": 2,
}

def normalize_exercise(val):
    return EXERCISE_MAP.get(str(val).strip().lower(), 0)

def predict_risk(age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate):
    X = np.array([[age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate]])
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])
    label = "High Risk" if pred == 1 else "Low Risk"
    return label, prob

# -------------------------------------------------
# Database Model
# -------------------------------------------------
class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.Integer, nullable=False)
    systolic_bp = db.Column(db.Float, nullable=False)
    diastolic_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    risk_label = db.Column(db.String(32), nullable=False)
    prob_high = db.Column(db.Float, nullable=False)

# -------------------------------------------------
# Create Tables (NO AUTO DROP)
# -------------------------------------------------
with app.app_context():
    db.create_all()

# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.get("/health")
def health():
    return jsonify(status="ok")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            age = float(request.form.get("age"))
            bmi = float(request.form.get("bmi"))
            exercise_level = normalize_exercise(request.form.get("exercise_level"))
            systolic_bp = float(request.form.get("systolic_bp"))
            diastolic_bp = float(request.form.get("diastolic_bp"))
            heart_rate = float(request.form.get("heart_rate"))

            label, prob_high = predict_risk(
                age, bmi, exercise_level,
                systolic_bp, diastolic_bp, heart_rate
            )

            # Save to database
            row = Prediction(
                age=age,
                bmi=bmi,
                exercise_level=exercise_level,
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                heart_rate=heart_rate,
                risk_label=label,
                prob_high=prob_high
            )

            db.session.add(row)
            db.session.commit()

            prediction = label
            probability = round(prob_high * 100, 2)

        except Exception as e:
            db.session.rollback()
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error
    )

@app.get("/history")
def history():
    records = Prediction.query.order_by(
        Prediction.created_at.desc()
    ).all()

    return render_template("history.html", records=records)

# -------------------------------------------------
# Debug Route (Remove later if you want)
# -------------------------------------------------
@app.get("/debug")
def debug():
    return {
        "db_uri": app.config["SQLALCHEMY_DATABASE_URI"],
        "count": Prediction.query.count()
    }

# -------------------------------------------------
# Run (Local Only)
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)`
