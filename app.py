from flask import Flask, render_template, request
import numpy as np
import joblib
import os

# --- DB (SQLite local, Postgres on deploy if DATABASE_URL is set) ---
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

app = Flask(__name__)

# -----------------------------
# Load model safely
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "demo_model.pkl")
model = joblib.load(MODEL_PATH)

# -----------------------------
# Database setup
# -----------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# If deployed (Render sets RENDER), require DATABASE_URL
if os.environ.get("RENDER") and not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required on Render")

# Local fallback only
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///predictions.db"

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    age = Column(Float)
    bmi = Column(Float)
    exercise = Column(Float)
    sys_bp = Column(Float)
    dia_bp = Column(Float)
    heart_rate = Column(Float)
    prediction = Column(String(20))
    probability = Column(Float)

Base.metadata.create_all(engine)

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
# app.py (MAKE THESE SMALL EDITS INSIDE home() ONLY)
# 1) Add: error = None
# 2) Add simple validation + pass error to template

# --- inside home() ---
def home():
    prediction_text = None
    probability_percent = None
    probability_class1_percent = None
    risk_class = None
    error = None

    form_vals = {
        "age": "", "bmi": "", "exercise": "", "sys_bp": "", "dia_bp": "", "heart_rate": ""
    }

    if request.method == "POST":
        try:
            age = float(request.form["age"])
            bmi = float(request.form["bmi"])
            exercise = float(request.form["exercise"])
            sys_bp = float(request.form["sys_bp"])
            dia_bp = float(request.form["dia_bp"])
            heart_rate = float(request.form["heart_rate"])

            form_vals = {
                "age": request.form["age"],
                "bmi": request.form["bmi"],
                "exercise": request.form["exercise"],
                "sys_bp": request.form["sys_bp"],
                "dia_bp": request.form["dia_bp"],
                "heart_rate": request.form["heart_rate"],
            }

            # Simple sanity checks
            if dia_bp >= sys_bp:
                raise ValueError("Diastolic BP must be lower than Systolic BP.")

            X = np.array([[age, bmi, exercise, sys_bp, dia_bp, heart_rate]])

            pred_class = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0]
            p0, p1 = float(proba[0]), float(proba[1])

            prediction_text = "High Risk" if pred_class == 1 else "Low Risk"
            risk_class = "high" if pred_class == 1 else "low"

            p_pred = p1 if pred_class == 1 else p0
            probability_percent = round(p_pred * 100, 2)
            probability_class1_percent = round(p1 * 100, 2)

            db = SessionLocal()
            try:
                db.add(PredictionLog(
                    age=age, bmi=bmi, exercise=exercise,
                    sys_bp=sys_bp, dia_bp=dia_bp, heart_rate=heart_rate,
                    prediction=prediction_text,
                    probability=probability_percent
                ))
                db.commit()
            finally:
                db.close()

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        prediction=prediction_text,
        probability=probability_percent,
        risk_meter=probability_class1_percent,
        risk_class=risk_class,
        form_vals=form_vals,
        error=error
)

@app.route("/history")
def history():
    db = SessionLocal()
    try:
        rows = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).limit(25).all()
    finally:
        db.close()

    return render_template("history.html", rows=rows)

if __name__ == "__main__":
    app.run(debug=True)
