import os
from datetime import datetime

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy

# ----------------------------
# App
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Database config (Render Postgres + local fallback)
# ----------------------------
db_url = os.getenv("DATABASE_URL", "").strip()

# Render often gives postgres:// but SQLAlchemy wants postgresql://
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

# Local fallback (only for local dev)
if not db_url:
    db_url = "sqlite:///local.db"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ----------------------------
# Model
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "demo_model.pkl")
model = joblib.load(MODEL_PATH)

FEATURE_ORDER = ["age", "bmi", "exercise_level", "systolic_bp", "diastolic_bp", "heart_rate"]

EXERCISE_MAP = {
    "low": 0,
    "moderate": 1,
    "high": 2,
    "0": 0,
    "1": 1,
    "2": 2,
}

def normalize_exercise(x) -> int:
    if x is None:
        return 0
    s = str(x).strip().lower()
    return EXERCISE_MAP.get(s, 0)

def predict_risk(age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate):
    # Model expects numeric array in the same order as training
    X = np.array([[age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate]], dtype=float)

    # Predict
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])  # class 1 probability

    label = "High Risk" if pred == 1 else "Low Risk"
    return label, proba

# ----------------------------
# DB Model
# ----------------------------
class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.Integer, nullable=False)  # 0/1/2
    systolic_bp = db.Column(db.Float, nullable=False)
    diastolic_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    risk_label = db.Column(db.String(32), nullable=False)
    prob_high = db.Column(db.Float, nullable=False)  # stored as 0..1


# ----------------------------
# Create tables (and optional RESET without shell)
# ----------------------------
with app.app_context():
    # IMPORTANT:
    # Set RESET_DB=1 ONE TIME on Render if your schema is mismatched.
    # Then set it back to 0 (or delete it) so you don't wipe history each deploy.
    if os.getenv("RESET_DB", "0") == "1":
        db.drop_all()
        db.create_all()
    else:
        db.create_all()

# ----------------------------
# Routes
# ----------------------------
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
            age = float(request.form.get("age", "0"))
            bmi = float(request.form.get("bmi", "0"))
            exercise_level = normalize_exercise(request.form.get("exercise_level", "low"))
            systolic_bp = float(request.form.get("systolic_bp", "0"))
            diastolic_bp = float(request.form.get("diastolic_bp", "0"))
            heart_rate = float(request.form.get("heart_rate", "0"))

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
                risk_label=label,
                prob_high=prob_high,
            )
            db.session.add(row)
            db.session.commit()

            prediction = label
            probability = round(prob_high * 100, 2)

        except Exception as e:
            # show a helpful error on the page
            db.session.rollback()
            error = f"{type(e).__name__}: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error,
    )

# API route for Instant (JS) mode (also saves to DB)
@app.post("/api/predict")
def api_predict():
    try:
        data = request.get_json(force=True) or {}

        age = float(data.get("age", 0))
        bmi = float(data.get("bmi", 0))
        exercise_level = normalize_exercise(data.get("exercise_level", "low"))
        systolic_bp = float(data.get("systolic_bp", 0))
        diastolic_bp = float(data.get("diastolic_bp", 0))
        heart_rate = float(data.get("heart_rate", 0))

        label, prob_high = predict_risk(
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
        )

        row = Prediction(
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
            risk_label=label,
            prob_high=prob_high,
        )
        db.session.add(row)
        db.session.commit()

        return jsonify(
            label=label,
            prob_high=prob_high,
            prob_percent=round(prob_high * 100, 2),
        )

    except Exception as e:
        db.session.rollback()
        return jsonify(error=f"{type(e).__name__}: {e}"), 500

@app.get("/history")
def history():
    records = (
        Prediction.query
        .order_by(Prediction.created_at.desc())
        .limit(200)
        .all()
    )
    return render_template("history.html", records=records)


# Render/Gunicorn looks for "app"
# (If you use wsqi.py, it's fine too, but this works directly.)
if __name__ == "__main__":
    app.run(debug=True)
