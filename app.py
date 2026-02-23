import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import joblib

app = Flask(__name__)

# ---------------- DATABASE ----------------

db_url = os.getenv("DATABASE_URL", "").strip()

if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

if not db_url:
    db_url = "sqlite:///local.db"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ---------------- MODEL ----------------

MODEL_PATH = "demo_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None


# ---------------- DATABASE TABLE ----------------

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.String(32), nullable=False)
    systolic_bp = db.Column(db.Float, nullable=False)
    diastolic_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    risk_label = db.Column(db.String(32), nullable=False)
    prob_high = db.Column(db.Float, nullable=False)


with app.app_context():
    db.drop_all()
    db.create_all()


# ---------------- ROUTES ----------------

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

        if model:
            exercise_map = {"Low": 0, "Moderate": 1, "High": 2}
            exercise_num = exercise_map.get(exercise_level, 0)

            data = np.array([[age, bmi, exercise_num, systolic_bp, diastolic_bp, heart_rate]])

            pred = int(model.predict(data)[0])
            prob_high = float(model.predict_proba(data)[0][1])
        else:
            # fallback dummy prediction
            pred = 0
            prob_high = 0.25

        prediction = "High Risk" if pred == 1 else "Low Risk"
        probability = round(prob_high * 100, 2)

        row = Prediction(
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
            risk_label=prediction,
            prob_high=prob_high,
        )

        db.session.add(row)
        db.session.commit()

    return render_template("index.html", prediction=prediction, probability=probability)


@app.get("/history")
def history():
    rows = (
        Prediction.query
        .order_by(Prediction.created_at.desc())
        .limit(200)
        .all()
    )

    return render_template("history.html", records=rows)


if __name__ == "__main__":
    app.run(debug=True)
