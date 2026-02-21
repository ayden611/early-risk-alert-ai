```python
import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from sqlalchemy import create_engine, text
import joblib

app = Flask(__name__)
app.secret_key = "dev-secret-key"

# =========================
# MODEL LOAD
# =========================
model = None
model_error = None

try:
    model = joblib.load("demo_model.pkl")
except Exception as e:
    model_error = str(e)

# =========================
# DATABASE SETUP
# =========================
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if DATABASE_URL:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
else:
    engine = create_engine("sqlite:///local.db")

# =========================
# FORCE CLEAN TABLE (DEV SAFE)
# =========================
def init_db():
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS predictions"))

        conn.execute(text("""
            CREATE TABLE predictions (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP,
                age FLOAT,
                bmi FLOAT,
                exercise_level TEXT,
                systolic_bp FLOAT,
                diastolic_bp FLOAT,
                heart_rate FLOAT,
                risk_label TEXT,
                probability FLOAT
            )
        """))

try:
    init_db()
except:
    pass

# =========================
# PREDICTION HELPER
# =========================
def predict_probability(age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate):
    if model is None:
        raise RuntimeError(model_error)

    mapping = {"Low": 0, "Moderate": 1, "High": 2}
    ex = mapping.get(exercise_level, 0)

    X = [[age, bmi, ex, systolic_bp, diastolic_bp, heart_rate]]
    return float(model.predict_proba(X)[0][1])

# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            age = float(request.form["age"])
            bmi = float(request.form["bmi"])
            exercise_level = request.form["exercise_level"]
            systolic_bp = float(request.form["systolic_bp"])
            diastolic_bp = float(request.form["diastolic_bp"])
            heart_rate = float(request.form["heart_rate"])

            proba = predict_probability(
                age, bmi, exercise_level,
                systolic_bp, diastolic_bp, heart_rate
            )

            label = "High" if proba >= 0.5 else "Low"

            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO predictions
                    (created_at, age, bmi, exercise_level,
                     systolic_bp, diastolic_bp,
                     heart_rate, risk_label, probability)
                    VALUES (:created_at, :age, :bmi, :exercise_level,
                            :systolic_bp, :diastolic_bp,
                            :heart_rate, :risk_label, :probability)
                """), {
                    "created_at": datetime.utcnow(),
                    "age": age,
                    "bmi": bmi,
                    "exercise_level": exercise_level,
                    "systolic_bp": systolic_bp,
                    "diastolic_bp": diastolic_bp,
                    "heart_rate": heart_rate,
                    "risk_label": label,
                    "probability": proba
                })

            return render_template("index.html", result={
                "label": label,
                "probability": round(proba * 100, 1)
            })

        except Exception as e:
            flash(str(e))
            return redirect(url_for("home"))

    return render_template("index.html")


@app.route("/history")
def history():
    rows = []
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT created_at, age, bmi, exercise_level,
                       systolic_bp, diastolic_bp,
                       heart_rate, risk_label, probability
                FROM predictions
                ORDER BY created_at DESC
                LIMIT 50
            """))
            rows = result.mappings().all()
    except:
        rows = []

    return render_template("history.html", rows=rows)


@app.route("/clear-history", methods=["POST"])
def clear_history():
    try:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM predictions"))
    except:
        pass
    return redirect(url_for("history"))


@app.route("/health")
def health():
    return {
        "ok": True,
        "model_loaded": model is not None,
        "db_connected": True
    }

if __name__ == "__main__":
    app.run(debug=True)
```
