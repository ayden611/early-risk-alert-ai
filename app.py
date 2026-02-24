import os
import datetime as dt
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# ==========================
# DATABASE CONFIG
# ==========================
db_url = os.getenv("DATABASE_URL")

# Fix for Render postgres (if needed)
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ==========================
# MODEL
# ==========================
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)

    age = db.Column(db.Integer, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.Integer, nullable=False)

    systolic_bp = db.Column(db.Integer, nullable=False)
    diastolic_bp = db.Column(db.Integer, nullable=False)
    heart_rate = db.Column(db.Integer, nullable=False)

    risk_label = db.Column(db.String(50), nullable=False)
    probability = db.Column(db.Float, nullable=False)


# ==========================
# SIMPLE RISK LOGIC
# ==========================
def calculate_risk(age, bmi, exercise_level, systolic_bp):
    score = 0

    if age > 50:
        score += 1
    if bmi > 30:
        score += 1
    if systolic_bp > 140:
        score += 1
    if exercise_level == 1:  # Low exercise
        score += 1

    if score >= 3:
        return "High", 0.85
    elif score == 2:
        return "Medium", 0.55
    else:
        return "Low", 0.15


EXERCISE_MAP = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}

# ==========================
# ROUTES
# ==========================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form.get("age"))
    bmi = float(request.form.get("bmi"))
    systolic_bp = int(request.form.get("systolic_bp"))
    diastolic_bp = int(request.form.get("diastolic_bp"))
    heart_rate = int(request.form.get("heart_rate"))

    exercise_str = request.form.get("exercise_level")
    exercise_level = EXERCISE_MAP.get(exercise_str, 1)

    risk_label, probability = calculate_risk(
        age, bmi, exercise_level, systolic_bp
    )

    prediction = Prediction(
        age=age,
        bmi=bmi,
        exercise_level=exercise_level,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        heart_rate=heart_rate,
        risk_label=risk_label,
        probability=probability
    )

    db.session.add(prediction)
    db.session.commit()

    return redirect(url_for("history"))


@app.route("/history")
def history():
    predictions = Prediction.query.order_by(
        Prediction.created_at.desc()
    ).all()

    return render_template(
        "history.html",
        predictions=predictions
    )


# ==========================
# CREATE TABLES
# ==========================
with app.app_context():
    db.create_all()


# ==========================
# RUN
# ==========================
if __name__ == "__main__":
    app.run(debug=True)
