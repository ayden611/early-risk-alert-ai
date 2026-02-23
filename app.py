import os
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy

from predict import predict_risk

app = Flask(__name__)

# =========================
# DATABASE CONFIG
# =========================

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # Render Postgres fix
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")
    app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///local.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# =========================
# MODEL TABLE
# =========================

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    age = db.Column(db.Float)
    bmi = db.Column(db.Float)
    exercise_level = db.Column(db.String(20))
    systolic_bp = db.Column(db.Float)
    diastolic_bp = db.Column(db.Float)
    heart_rate = db.Column(db.Float)

    label = db.Column(db.String(20))
    probability = db.Column(db.Float)


# Create tables automatically
with app.app_context():
    db.create_all()


# =========================
# ROUTES
# =========================

@app.get("/health")
def health():
    return jsonify(status="ok")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":
        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        exercise_level = request.form["exercise_level"]
        systolic_bp = float(request.form["systolic_bp"])
        diastolic_bp = float(request.form["diastolic_bp"])
        heart_rate = float(request.form["heart_rate"])

        label, prob_high = predict_risk(
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
        )

        prediction = label
        probability = round(prob_high * 100, 2)

        # =========================
        # SAVE TO DATABASE
        # =========================
        record = Prediction(
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
            label=label,
            probability=prob_high,
        )

        db.session.add(record)
        db.session.commit()

    return render_template("index.html", prediction=prediction, probability=probability)


@app.route("/history")
def history():
    records = Prediction.query.order_by(Prediction.timestamp.desc()).limit(50).all()
    return render_template("history.html", records=records)


if __name__ == "__main__":
    app.run(debug=True)
