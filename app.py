import os
from datetime import datetime

import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify, render_template_string
from flask_sqlalchemy import SQLAlchemy

# --------------------------------------------------
# App Setup
# --------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev")

db_url = os.getenv("DATABASE_URL")

# Render sometimes uses postgres:// which SQLAlchemy doesn't accept
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

# Fallback local DB
if not db_url:
    db_url = "sqlite:///local.db"

app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "demo_model.pkl")
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

FEATURES = ["age", "bmi", "exercise_level", "systolic_bp", "diastolic_bp", "heart_rate"]

# --------------------------------------------------
# DB Model
# --------------------------------------------------
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


with app.app_context():
    db.create_all()

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def normalize_exercise(x) -> str:
    if x is None:
        return "low"
    s = str(x).strip().lower()
    if s in ("0", "low", "l"):
        return "low"
    if s in ("1", "moderate", "med", "m"):
        return "moderate"
    if s in ("2", "high", "h"):
        return "high"
    # if user typed anything else, default safe
    return "low"


def exercise_to_num(level: str) -> float:
    level = normalize_exercise(level)
    return {"low": 0.0, "moderate": 1.0, "high": 2.0}.get(level, 0.0)


def predict_risk(age, bmi, exercise_level, systolic_bp, diastolic_bp, heart_rate):
    # If model missing, return a safe placeholder
    if model is None:
        return "Model Not Loaded", 0.0

    ex_num = exercise_to_num(exercise_level)

    X = np.array([[age, bmi, ex_num, systolic_bp, diastolic_bp, heart_rate]], dtype=float)
    pred = int(model.predict(X)[0])

    # proba for class 1 (high risk) if available
    prob_high = 0.0
    if hasattr(model, "predict_proba"):
        prob_high = float(model.predict_proba(X)[0][1])

    label = "High Risk" if pred == 1 else "Low Risk"
    return label, prob_high


def template_exists(path: str) -> bool:
    base = os.path.dirname(__file__)
    return os.path.exists(os.path.join(base, "templates", path))


# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/health")
def health():
    return jsonify(status="ok")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            age = float(request.form.get("age", "0"))
            bmi = float(request.form.get("bmi", "0"))
            exercise_level = normalize_exercise(request.form.get("exercise_level", "low"))
            systolic_bp = float(request.form.get("systolic_bp", "0"))
            diastolic_bp = float(request.form.get("diastolic_bp", "0"))
            heart_rate = float(request.form.get("heart_rate", "0"))

            label, prob_high = predict_risk(
                age=age,
                bmi=bmi,
                exercise_level=exercise_level,
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                heart_rate=heart_rate,
            )

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

            prediction = label
            probability = round(prob_high * 100, 2)

        except Exception as e:
            db.session.rollback()
            error = f"{type(e).__name__}: {e}"

    # Use your template if present, else fallback HTML
    if template_exists("index.html"):
        return render_template(
            "index.html",
            prediction=prediction,
            probability=probability,
            error=error,
        )

    # Fallback minimal page
    return render_template_string(
        """
        <!doctype html>
        <html>
        <head><meta charset="utf-8"><title>Early Risk Alert AI</title></head>
        <body style="font-family:system-ui;margin:30px;max-width:720px">
          <h1>Early Risk Alert AI</h1>
          <form method="post">
            <label>Age <input name="age" type="number" step="0.1" required></label><br><br>
            <label>BMI <input name="bmi" type="number" step="0.1" required></label><br><br>
            <label>Exercise Level
              <select name="exercise_level">
                <option value="low">Low</option>
                <option value="moderate">Moderate</option>
                <option value="high">High</option>
              </select>
            </label><br><br>
            <label>Systolic BP <input name="systolic_bp" type="number" step="0.1" required></label><br><br>
            <label>Diastolic BP <input name="diastolic_bp" type="number" step="0.1" required></label><br><br>
            <label>Heart Rate <input name="heart_rate" type="number" step="0.1" required></label><br><br>
            <button type="submit">Run Prediction</button>
          </form>

          <p><a href="/history">View History</a></p>

          {% if error %}
            <p style="color:#b00020;"><b>Error:</b> {{error}}</p>
          {% endif %}

          {% if prediction %}
            <h2>Result</h2>
            <p><b>Risk:</b> {{prediction}}</p>
            <p><b>Probability:</b> {{probability}}%</p>
          {% endif %}
        </body>
        </html>
        """,
        prediction=prediction,
        probability=probability,
        error=error,
    )


@app.get("/history")
def history():
    rows = (
        Prediction.query.order_by(Prediction.created_at.desc())
        .limit(200)
        .all()
    )

    if template_exists("history.html"):
        return render_template("history.html", records=rows)

    # Fallback minimal history page
    return render_template_string(
        """
        <!doctype html>
        <html>
        <head><meta charset="utf-8"><title>History</title></head>
        <body style="font-family:system-ui;margin:30px;max-width:900px">
          <h1>Prediction History</h1>
          <p><a href="/">Back</a></p>

          {% if not rows %}
            <p>No history yet.</p>
          {% else %}
            <table border="1" cellpadding="8" cellspacing="0">
              <tr>
                <th>Date</th><th>Age</th><th>BMI</th><th>Exercise</th>
                <th>Sys</th><th>Dia</th><th>HR</th>
                <th>Result</th><th>Prob High</th>
              </tr>
              {% for r in rows %}
                <tr>
                  <td>{{r.created_at}}</td>
                  <td>{{"%.1f"|format(r.age)}}</td>
                  <td>{{"%.1f"|format(r.bmi)}}</td>
                  <td>{{r.exercise_level}}</td>
                  <td>{{"%.1f"|format(r.systolic_bp)}}</td>
                  <td>{{"%.1f"|format(r.diastolic_bp)}}</td>
                  <td>{{"%.1f"|format(r.heart_rate)}}</td>
                  <td>{{r.risk_label}}</td>
                  <td>{{"%.4f"|format(r.prob_high)}}</td>
                </tr>
              {% endfor %}
            </table>
          {% endif %}
        </body>
        </html>
        """,
        rows=rows,
    )


# API route for Instant (JS) mode (also saves to DB)
@app.post("/api/predict")
def api_predict():
    try:
        data = request.get_json(force=True) or {}

        age = float(data.get("age", 0))
        bmi = float(data.get("bmi", 0))
        exercise_level = normalize_exercise(data.get("exercise_level", "low"))
        systolic_bp = float(data.get("systolic_bp", 0))
        diastolic_bp = float(data.get("diastolic_bp", 0))
        heart_rate = float(data.get("heart_rate", 0))

        label, prob_high = predict_risk(
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
        )

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

        return jsonify(
            prediction=label,
            prob_high=float(prob_high),
            probability=round(float(prob_high) * 100, 2),
        )

    except Exception as e:
        db.session.rollback()
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


if __name__ == "__main__":
    # Local run only (Render uses gunicorn/wsgi)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)`
