import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# --- Database config (Render provides DATABASE_URL) ---
db_url = os.environ.get("DATABASE_URL")

# Render sometimes gives postgres:// which SQLAlchemy wants as postgresql://
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

# If DATABASE_URL isn't set locally, fall back to sqlite
app.config["SQLALCHEMY_DATABASE_URI"] = db_url or "sqlite:///local.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# --- Model ---
class Prediction(db.Model):
    __tablename__ = "prediction"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    age = db.Column(db.Integer, nullable=False)
    bmi = db.Column(db.Float, nullable=False)

    # store numeric level: 1=Low, 2=Medium, 3=High
    exercise_level = db.Column(db.Integer, nullable=False)

    systolic_bp = db.Column(db.Integer, nullable=False)
    diastolic_bp = db.Column(db.Integer, nullable=False)
    heart_rate = db.Column(db.Integer, nullable=False)

    risk_label = db.Column(db.String(50), nullable=False)
    probability = db.Column(db.Float, nullable=False)

# Create tables on startup (safe for small apps; for bigger apps use migrations)
with app.app_context():
    db.create_all()

# --- Helpers ---
EXERCISE_MAP = {"Low": 1, "Medium": 2, "High": 3}
EXERCISE_LABEL = {1: "Low", 2: "Medium", 3: "High"}

def safe_int(val, default=0):
    try:
        return int(val)
    except Exception:
        return default

def safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default

# --- Routes ---
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = safe_int(request.form.get("age"))
    bmi = safe_float(request.form.get("bmi"))

    exercise_str = request.form.get("exercise_level", "Low")
    exercise_level = EXERCISE_MAP.get(exercise_str, 1)

    systolic_bp = safe_int(request.form.get("systolic_bp"))
    diastolic_bp = safe_int(request.form.get("diastolic_bp"))
    heart_rate = safe_int(request.form.get("heart_rate"))

    probability = 0.9 if (systolic_bp > 140 or diastolic_bp > 90) else 0.2
    risk_label = "High Risk" if probability >= 0.5 else "Low Risk"

    p = Prediction(
        created_at=datetime.utcnow(),
        age=age,
        bmi=bmi,
        exercise_level=exercise_level,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        heart_rate=heart_rate,
        risk_label=risk_label,
        probability=float(probability),
    )
    db.session.add(p)
    db.session.commit()

    return render_template(
        "index.html",
        result=risk_label,
        probability=round(probability * 100, 2),
        form_data={
            "age": age,
            "bmi": bmi,
            "exercise_level": exercise_str,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "heart_rate": heart_rate,
        },
    )

@app.route("/history", methods=["GET"])
def history():
    predictions = (
        Prediction.query.order_by(Prediction.created_at.desc())
        .limit(50)
        .all()
    )
    return render_template("history.html", predictions=predictions, exercise_label=EXERCISE_LABEL)

if __name__ == "__main__":
    app.run(debug=True)
