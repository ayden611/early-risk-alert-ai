import os
from datetime import datetime
from types import SimpleNamespace

import joblib
import numpy as np
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text, inspect


# ----------------------------
# App setup
# ----------------------------
app = Flask(__name__)

# Secrets / Auth
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "password")

# Database
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///predictions.db")

# Render Postgres sometimes gives postgres:// (SQLAlchemy wants postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "demo_model.pkl")
model = joblib.load(MODEL_PATH)


# ----------------------------
# DB Model
# ----------------------------
class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.String(20), nullable=False)

    systolic_bp = db.Column(db.Float, nullable=False)
    diastolic_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    risk_label = db.Column(db.String(20), nullable=False)
    probability = db.Column(db.Float, nullable=False)  # 0..1


def _ensure_schema():
    """
    Creates tables if missing and adds missing columns if your DB existed already
    with an older schema (common on Render).
    """
    db.create_all()

    try:
        insp = inspect(db.engine)
        cols = {c["name"] for c in insp.get_columns("predictions")}

        # If the table exists but a column is missing, add it safely.
        alter_stmts = []
        if "exercise_level" not in cols:
            alter_stmts.append("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS exercise_level VARCHAR(20);")
        if "risk_label" not in cols:
            alter_stmts.append("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS risk_label VARCHAR(20);")
        if "probability" not in cols:
            alter_stmts.append("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS probability DOUBLE PRECISION;")

        if alter_stmts:
            with db.engine.begin() as conn:
                for stmt in alter_stmts:
                    conn.execute(text(stmt))
    except Exception:
        pass


with app.app_context():
    _ensure_schema()


# ----------------------------
# Helpers
# ----------------------------
def login_required(fn):
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


def exercise_to_num(level: str) -> float:
    level = (level or "").strip().lower()
    if level == "low":
        return 0.0
    if level == "moderate":
        return 1.0
    if level == "high":
        return 2.0
    return 0.0


def make_form_ns(**kwargs):
    defaults = dict(
        age="",
        bmi="",
        exercise_level="Low",
        systolic_bp="",
        diastolic_bp="",
        heart_rate="",
        acknowledged=False,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ----------------------------
# Routes
# ----------------------------
@app.route("/health")
def health():
    return {"status": "ok"}, 200


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("home"))

        flash("Invalid username or password.")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    prediction = None
    probability_pct = None

    form = make_form_ns()

    if request.method == "POST":
        try:
            acknowledged = request.form.get("acknowledged") == "on"

            age = float(request.form.get("age", "0"))
            bmi = float(request.form.get("bmi", "0"))
            exercise_level = request.form.get("exercise_level", "Low")
            systolic_bp = float(request.form.get("systolic_bp", "0"))
            diastolic_bp = float(request.form.get("diastolic_bp", "0"))
            heart_rate = float(request.form.get("heart_rate", "0"))

            form = make_form_ns(
                age=age,
                bmi=bmi,
                exercise_level=exercise_level,
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                heart_rate=heart_rate,
                acknowledged=acknowledged,
            )

            if not acknowledged:
                flash("Please acknowledge the disclaimer checkbox before predicting.")
                return render_template("index.html", form=form, prediction=None, probability=None)

            ex_num = exercise_to_num(exercise_level)

            X = np.array([[age, bmi, ex_num, systolic_bp, diastolic_bp, heart_rate]], dtype=float)

            pred = int(model.predict(X)[0])
            prob_high = float(model.predict_proba(X)[0][1])

            prediction = "High" if pred == 1 else "Low"
            probability_pct = round(prob_high * 100.0, 1)

            row = Prediction(
                age=age,
                bmi=bmi,
                exercise_level=exercise_level,
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                heart_rate=heart_rate,
                risk_label=prediction,
                probability=prob_high,
            )
            db.session.add(row)
            db.session.commit()

        except Exception as e:
            db.session.rollback()
            flash(f"Error: {e}")

    return render_template(
        "index.html",
        form=form,
        prediction=prediction,
        probability=probability_pct,
    )


@app.route("/history", methods=["GET"])
@login_required
def history():
    rows = Prediction.query.order_by(Prediction.created_at.desc()).limit(50).all()
    return render_template("history.html", rows=rows)


@app.route("/history/clear", methods=["POST"])
@login_required
def clear_history():
    try:
        Prediction.query.delete()
        db.session.commit()
        flash("History cleared.")
    except Exception as e:
        db.session.rollback()
        flash(f"Could not clear history: {e}")
    return redirect(url_for("history"))
