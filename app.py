import os
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy

from predict import predict_risk

app = Flask(__name__)

# ---------- DATABASE CONFIG ----------
# Render Postgres provides DATABASE_URL. Locally we fallback to sqlite.
db_url = os.getenv("DATABASE_URL", "").strip()

# Render sometimes uses postgres:// which SQLAlchemy wants as postgresql://
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

if not db_url:
    db_url = "sqlite:///local.db"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ---------- MODEL ----------
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.String(32), nullable=False)
    systolic_bp = db.Column(db.Float, nullable=False)
    diastolic_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    risk_label = db.Column(db.String(32), nullable=False)
    prob_high = db.Column(db.Float, nullable=False)  # 0..1


# Create tables at startup (prevents "table doesn't exist" crashes)
with app.app_context():
    db.create_all()


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

        # Save to DB
        row = Prediction(
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
            risk_label=label,
            prob_high=float(prob_high),
        )
        db.session.add(row)
        db.session.commit()

    return render_template("index.html", prediction=prediction, probability=probability)


@app.get("/history")
def history():
    # If something goes wrong, show the error text (TEMP for debugging)
    try:
        rows = (
            Prediction.query.order_by(Prediction.created_at.desc())
            .limit(200)
            .all()
        )
        return render_template("history.html", rows=rows)
    except Exception as e:
        return f"History error: {e}", 500
