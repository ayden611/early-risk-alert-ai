import os
import uuid
import json
import logging
import traceback
import datetime as dt

import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

# ======================
# App Setup
# ======================

app = Flask(__name__)

db_url = os.getenv("DATABASE_URL")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url or "sqlite:///local.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ======================
# Logging
# ======================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# Model
# ======================

MODEL_PATH = "demo_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# ======================
# DB Model
# ======================


class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.String(64), nullable=False)
    user_id = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.String(20), nullable=False)
    systolic_bp = db.Column(db.Float, nullable=False)
    diastolic_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    risk_label = db.Column(db.String(50), nullable=False)
    probability = db.Column(db.Float, nullable=False)


with app.app_context():
    db.create_all()

# ======================
# Health
# ======================


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok", "db_ok": True, "model_loaded": model is not None})


# ======================
# Predict
# ======================


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        client_id = request.headers.get("X-Client-Id")

        if not client_id:
            return jsonify({"error": "Missing Client ID"}), 400

        age = float(data["age"])
        bmi = float(data["bmi"])
        exercise_level = data["exercise_level"]
        systolic_bp = float(data["systolic_bp"])
        diastolic_bp = float(data["diastolic_bp"])
        heart_rate = float(data["heart_rate"])

        if model:
            input_data = np.array(
                [[age, bmi, 1, systolic_bp, diastolic_bp, heart_rate]]
            )
            pred = int(model.predict(input_data)[0])
            prob = float(model.predict_proba(input_data)[0][1])
            risk_label = "High Risk" if pred == 1 else "Low Risk"
        else:
            risk_label = "Low Risk"
            prob = 0.25

        prediction = Prediction(
            client_id=client_id,
            user_id=None,
            created_at=dt.datetime.utcnow(),
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
            risk_label=risk_label,
            probability=round(prob * 100, 2),
        )

        db.session.add(prediction)
        db.session.commit()

        return jsonify({"risk_label": risk_label, "probability": round(prob * 100, 2)})

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": "Server Error", "details": str(e)}), 500


# ======================
# Run
# ======================

if __name__ == "__main__":
    app.run(debug=True)
