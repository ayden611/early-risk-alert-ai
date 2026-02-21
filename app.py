import os
from datetime import datetime
import csv
import io

import joblib
import numpy as np
from flask import (
    Flask, render_template, request,
    redirect, url_for, session,
    flash, send_file
)
from flask_sqlalchemy import SQLAlchemy

# --------------------------------
# App Setup
# --------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "password")

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///predictions.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# --------------------------------
# Model
# --------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "demo_model.pkl")
model = joblib.load(MODEL_PATH)

# --------------------------------
# Database Model
# --------------------------------
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    age = db.Column(db.Float)
    bmi = db.Column(db.Float)
    exercise_level = db.Column(db.String(20))
    systolic_bp = db.Column(db.Float)
    diastolic_bp = db.Column(db.Float)
    heart_rate = db.Column(db.Float)

    risk_label = db.Column(db.String(20))
    probability = db.Column(db.Float)

with app.app_context():
    db.drop_all()
    db.create_all()

# --------------------------------
# Auth Helper
# --------------------------------
def login_required(fn):
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper

# --------------------------------
# Routes
# --------------------------------
@app.route("/health")
def health():
    return {"status": "ok"}, 200

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if (
            request.form.get("username") == ADMIN_USERNAME and
            request.form.get("password") == ADMIN_PASSWORD
        ):
            session["logged_in"] = True
            return redirect(url_for("home"))
        flash("Invalid credentials")
        return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# --------------------------------
# Main Prediction Page
# --------------------------------
@app.route("/", methods=["GET", "POST"])
@login_required
def home():

    form = {
        "age": "",
        "bmi": "",
        "exercise_level": "Low",
        "systolic_bp": "",
        "diastolic_bp": "",
        "heart_rate": "",
        "ack": False
    }

    prediction = None
    probability_pct = None

    if request.method == "POST":
        try:
            age = float(request.form["age"])
            bmi = float(request.form["bmi"])
            exercise_level = request.form["exercise_level"]
            systolic_bp = float(request.form["systolic_bp"])
            diastolic_bp = float(request.form["diastolic_bp"])
            heart_rate = float(request.form["heart_rate"])
            ack = request.form.get("ack") == "on"

            form.update({
                "age": age,
                "bmi": bmi,
                "exercise_level": exercise_level,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "heart_rate": heart_rate,
                "ack": ack
            })

            if not ack:
                flash("Please acknowledge the disclaimer.")
                return render_template("index.html",
                                       form=form,
                                       prediction=None,
                                       probability_pct=None)

            ex_map = {"Low": 0, "Moderate": 1, "High": 2}
            ex_val = ex_map.get(exercise_level, 0)

            X = np.array([[age, bmi, ex_val,
                           systolic_bp, diastolic_bp,
                           heart_rate]])

            pred = int(model.predict(X)[0])
            prob_high = float(model.predict_proba(X)[0][1])

            prediction = "High" if pred == 1 else "Low"
            probability_pct = round(prob_high * 100, 1)

            row = Prediction(
                age=age,
                bmi=bmi,
                exercise_level=exercise_level,
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                heart_rate=heart_rate,
                risk_label=prediction,
                probability=prob_high
            )

            db.session.add(row)
            db.session.commit()

        except Exception as e:
            db.session.rollback()
            flash(str(e))

    return render_template("index.html",
                           form=form,
                           prediction=prediction,
                           probability_pct=probability_pct)

# --------------------------------
# History
# --------------------------------
@app.route("/history")
@login_required
def history():
    rows = Prediction.query.order_by(
        Prediction.created_at.desc()
    ).all()
    return render_template("history.html", rows=rows)

@app.route("/clear_history", methods=["POST"])
@login_required
def clear_history():
    Prediction.query.delete()
    db.session.commit()
    return redirect(url_for("history"))

@app.route("/delete_row/<int:row_id>", methods=["POST"])
@login_required
def delete_row(row_id):
    row = Prediction.query.get_or_404(row_id)
    db.session.delete(row)
    db.session.commit()
    return redirect(url_for("history"))

# --------------------------------
# CSV Download
# --------------------------------
@app.route("/download_history_csv")
@login_required
def download_history_csv():
    rows = Prediction.query.order_by(
        Prediction.created_at.desc()
    ).all()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Time", "Age", "BMI", "Exercise",
        "Systolic", "Diastolic", "Heart Rate",
        "Risk", "Probability"
    ])

    for r in rows:
        writer.writerow([
            r.created_at,
            r.age,
            r.bmi,
            r.exercise_level,
            r.systolic_bp,
            r.diastolic_bp,
            r.heart_rate,
            r.risk_label,
            round(r.probability * 100, 1)
        ])

    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="prediction_history.csv"
    )
