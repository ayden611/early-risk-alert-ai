import os
import numpy as np
import joblib
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy

# --------------------------------------------------
# App Setup
# --------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev")

# Database
db_url = os.getenv("DATABASE_URL")

if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

if not db_url:
    db_url = "sqlite:///local.db"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

MODEL_PATH = "demo_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# --------------------------------------------------
# Database Model
# --------------------------------------------------

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    age = db.Column(db.Float)
    bmi = db.Column(db.Float)
    exercise_level = db.Column(db.Float)
    sys_bp = db.Column(db.Float)
    dia_bp = db.Column(db.Float)
    heart_rate = db.Column(db.Float)
    result = db.Column(db.String(50))
    probability = db.Column(db.Float)

# Create tables automatically
with app.app_context():
    db.create_all()

# --------------------------------------------------
# Helper Function
# --------------------------------------------------

def run_model(age, bmi, exercise_level, sys_bp, dia_bp, heart_rate):
    if model is None:
        return "Model Not Loaded", 0.0

    X = np.array([[age, bmi, exercise_level, sys_bp, dia_bp, heart_rate]])
    pred = int(model.predict(X)[0])
    prob_high = float(model.predict_proba(X)[0][1])

    prediction = "High Risk" if pred == 1 else "Low Risk"
    probability = round(prob_high * 100, 2)

    return prediction, probability

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":
        try:
            age = float(request.form["age"])
            bmi = float(request.form["bmi"])
            exercise_level = float(request.form["exercise_level"])
            sys_bp = float(request.form["sys_bp"])
            dia_bp = float(request.form["dia_bp"])
            heart_rate = float(request.form["heart_rate"])

            prediction, probability = run_model(
                age, bmi, exercise_level, sys_bp, dia_bp, heart_rate
            )

            new_entry = Prediction(
                age=age,
                bmi=bmi,
                exercise_level=exercise_level,
                sys_bp=sys_bp,
                dia_bp=dia_bp,
                heart_rate=heart_rate,
                result=prediction,
                probability=probability
            )

            db.session.add(new_entry)
            db.session.commit()

        except Exception as e:
            prediction = f"Error: {str(e)}"
            probability = 0

    return render_template("index.html",
                           prediction=prediction,
                           probability=probability)

# --------------------------------------------------
# Instant JS API
# --------------------------------------------------

@app.post("/api/predict")
def api_predict():
    try:
        data = request.get_json(force=True)

        age = float(data["age"])
        bmi = float(data["bmi"])
        exercise_level = float(data["exercise_level"])
        sys_bp = float(data["sys_bp"])
        dia_bp = float(data["dia_bp"])
        heart_rate = float(data["heart_rate"])

        prediction, probability = run_model(
            age, bmi, exercise_level, sys_bp, dia_bp, heart_rate
        )

        new_entry = Prediction(
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            sys_bp=sys_bp,
            dia_bp=dia_bp,
            heart_rate=heart_rate,
            result=prediction,
            probability=probability
        )

        db.session.add(new_entry)
        db.session.commit()

        return jsonify({
            "prediction": prediction,
            "probability": probability
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# History Page
# --------------------------------------------------

@app.route("/history")
def history():
    records = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    return render_template("history.html", records=records)

# --------------------------------------------------
# Render Port Binding
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
