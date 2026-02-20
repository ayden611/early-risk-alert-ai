import os
from datetime import datetime

import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash

from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")  # change in Render env

# ----------------------------
# Database (SQLite local, Postgres on Render)
# ----------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")

# Render often provides postgres URLs that start with postgres://
# SQLAlchemy wants postgresql://
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL or "sqlite:///predictions.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise = db.Column(db.Float, nullable=False)
    sys_bp = db.Column(db.Float, nullable=False)
    dia_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    label = db.Column(db.String(20), nullable=False)      # "High Risk" / "Low Risk"
    probability = db.Column(db.Float, nullable=False)     # 0-100


# Create tables (safe to call on startup)
with app.app_context():
    db.create_all()

# ----------------------------
# Load model safely
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "demo_model.pkl")
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    # Default values so Jinja never crashes
    form_vals = {
        "age": "",
        "bmi": "",
        "exercise": "",
        "sys_bp": "",
        "dia_bp": "",
        "heart_rate": "",
    }

    if request.method == "POST":
        try:
            form_vals["age"] = request.form.get("age", "")
            form_vals["bmi"] = request.form.get("bmi", "")
            form_vals["exercise"] = request.form.get("exercise", "")
            form_vals["sys_bp"] = request.form.get("sys_bp", "")
            form_vals["dia_bp"] = request.form.get("dia_bp", "")
            form_vals["heart_rate"] = request.form.get("heart_rate", "")

            age = float(form_vals["age"])
            bmi = float(form_vals["bmi"])
            exercise = float(form_vals["exercise"])
            sys_bp = float(form_vals["sys_bp"])
            dia_bp = float(form_vals["dia_bp"])
            heart_rate = float(form_vals["heart_rate"])

            data = np.array([[age, bmi, exercise, sys_bp, dia_bp, heart_rate]])

            pred = int(model.predict(data)[0])
            prob_high = float(model.predict_proba(data)[0][1])

            prediction = "High Risk" if pred == 1 else "Low Risk"
            probability = round(prob_high * 100, 2)

            # Save prediction
            row = Prediction(
                age=age,
                bmi=bmi,
                exercise=exercise,
                sys_bp=sys_bp,
                dia_bp=dia_bp,
                heart_rate=heart_rate,
                label=prediction,
                probability=probability,
            )
            db.session.add(row)
            db.session.commit()

        except Exception:
            db.session.rollback()
            flash("Please enter valid numbers in all fields.", "error")
            return redirect(url_for("home"))

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        form_vals=form_vals,
    )


@app.route("/history")
def history():
    recent = Prediction.query.order_by(Prediction.created_at.desc()).limit(25).all()
    return render_template("history.html", rows=recent)


if __name__ == "__main__":
    app.run(debug=True)
