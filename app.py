from flask import Flask, render_template, request
import joblib
import numpy as np
import os
from sqlalchemy import create_engine, text

app = Flask(__name__)

# ===============================
# DATABASE SETUP
# ===============================

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

def init_db():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                age FLOAT,
                bmi FLOAT,
                exercise_level TEXT,
                systolic_bp FLOAT,
                diastolic_bp FLOAT,
                heart_rate FLOAT,
                risk_label TEXT,
                risk_class TEXT,
                probability FLOAT,
                confidence FLOAT
            );
        """))

init_db()

# ===============================
# LOAD MODEL
# ===============================

model = joblib.load("demo_model.pkl")

# ===============================
# HOME ROUTE
# ===============================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None
    error = None

    form = {
        "age": "",
        "bmi": "",
        "exercise_level": "",
        "systolic_bp": "",
        "diastolic_bp": "",
        "heart_rate": "",
        "ack": False
    }

    if request.method == "POST":
        try:
            age = float(request.form["age"])
            bmi = float(request.form["bmi"])
            exercise_level = request.form["exercise_level"]
            systolic_bp = float(request.form["systolic_bp"])
            diastolic_bp = float(request.form["diastolic_bp"])
            heart_rate = float(request.form["heart_rate"])
            ack = request.form.get("ack") == "on"

            form.update(request.form)

            exercise_map = {
                "0 - Low": 0,
                "1 - Moderate": 1,
                "2 - High": 2
            }

            features = np.array([[ 
                age,
                bmi,
                exercise_map.get(exercise_level, 0),
                systolic_bp,
                diastolic_bp,
                heart_rate
            ]])

            probability = float(model.predict_proba(features)[0][1])
            confidence = round(probability * 100, 1)

            risk_class = "high" if probability >= 0.5 else "low"
            risk_label = "High Risk" if probability >= 0.5 else "Low Risk"

            result = {
                "risk_label": risk_label,
                "risk_class": risk_class,
                "probability": round(probability * 100, 1),
                "confidence": confidence
            }

            # SAVE TO DATABASE
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO predictions
                    (age,bmi,exercise_level,systolic_bp,diastolic_bp,heart_rate,
                     risk_label,risk_class,probability,confidence)
                    VALUES
                    (:age,:bmi,:exercise_level,:systolic_bp,:diastolic_bp,:heart_rate,
                     :risk_label,:risk_class,:probability,:confidence)
                """), {
                    "age": age,
                    "bmi": bmi,
                    "exercise_level": exercise_level,
                    "systolic_bp": systolic_bp,
                    "diastolic_bp": diastolic_bp,
                    "heart_rate": heart_rate,
                    "risk_label": risk_label,
                    "risk_class": risk_class,
                    "probability": round(probability * 100, 1),
                    "confidence": confidence
                })

        except Exception as e:
            error = "Invalid input. Please check your values."

    return render_template("index.html", result=result, error=error, form=form)


# ===============================
# HISTORY ROUTE
# ===============================

@app.route("/history")
def history():

    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT *
            FROM predictions
            ORDER BY created_at DESC
            LIMIT 30
        """)).mappings().all()

    return render_template("history.html", history=rows)


# ===============================
# RUN
# ===============================

if __name__ == "__main__":
    app.run(debug=
