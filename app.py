import os
from datetime import datetime, timezone, date
import io
import csv
import math

import joblib
import numpy as np
from flask import (
    Flask, render_template, request,
    redirect, url_for, session, flash, send_file
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func

# ----------------------------
# App Setup
# ----------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "password")

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///predictions.db")
# Render sometimes gives "postgres://", SQLAlchemy wants "postgresql://"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ----------------------------
# Load Model
# ----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "demo_model.pkl")
model = joblib.load(MODEL_PATH)

# ----------------------------
# DB Model
# ----------------------------
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    age = db.Column(db.Float)
    bmi = db.Column(db.Float)
    exercise_level = db.Column(db.String(20))
    systolic_bp = db.Column(db.Float)
    diastolic_bp = db.Column(db.Float)
    heart_rate = db.Column(db.Float)

    risk_label = db.Column(db.String(20))
    probability = db.Column(db.Float)  # stored as 0..1


with app.app_context():
    db.create_all()

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

def _to_float(v, default=None):
    try:
        return float(v)
    except Exception:
        return default

def _smooth_probability(p: float, temperature: float = 2.0, floor: float = 0.02, ceil: float = 0.98) -> float:
    """
    (1) QUICK PROBABILITY FIX:
    - Prevents extreme 0%/100% without retraining.
    - Uses "temperature scaling" on the logit.
    """
    # guard
    if p is None or math.isnan(p):
        return 0.5
    p = min(max(p, 1e-6), 1 - 1e-6)  # avoid log(0)
    logit = math.log(p / (1 - p))
    logit_scaled = logit / max(temperature, 1e-6)
    p2 = 1 / (1 + math.exp(-logit_scaled))
    # clip
    return float(min(max(p2, floor), ceil))

def _risk_from_prob(p: float, threshold: float = 0.5) -> str:
    return "High" if p >= threshold else "Low"

# ----------------------------
# Routes
# ----------------------------
@app.route("/health")
def health():
    return {"status": "ok"}, 200

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username", "")
        pw = request.form.get("password", "")
        if u == ADMIN_USERNAME and pw == ADMIN_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("home"))
        flash("Invalid credentials")
        return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# Main predictor
@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    prediction = None
    probability_pct = None

    # keep form sticky
    form = {
        "age": request.form.get("age", ""),
        "bmi": request.form.get("bmi", ""),
        "exercise_level": request.form.get("exercise_level", "Low"),
        "systolic_bp": request.form.get("systolic_bp", ""),
        "diastolic_bp": request.form.get("diastolic_bp", ""),
        "heart_rate": request.form.get("heart_rate", ""),
        "ack": request.form.get("ack", ""),
    }

    if request.method == "POST":
        # basic validation
        if not request.form.get("ack"):
            flash("Please confirm you understand this is not medical advice.")
            return render_template("index.html", prediction=None, probability_pct=None, form=form)

        age = _to_float(request.form.get("age"))
        bmi = _to_float(request.form.get("bmi"))
        systolic_bp = _to_float(request.form.get("systolic_bp"))
        diastolic_bp = _to_float(request.form.get("diastolic_bp"))
        heart_rate = _to_float(request.form.get("heart_rate"))
        exercise_level = request.form.get("exercise_level", "Low")

        # map exercise
        ex_map = {"Low": 0.0, "Moderate": 1.0, "High": 2.0}
        ex_num = ex_map.get(exercise_level, 0.0)

        if None in (age, bmi, systolic_bp, diastolic_bp, heart_rate):
            flash("Please enter valid numeric values.")
            return render_template("index.html", prediction=None, probability_pct=None, form=form)

        X = np.array([[age, bmi, ex_num, systolic_bp, diastolic_bp, heart_rate]], dtype=float)

        # model probability (class 1 => High)
        raw_p = float(model.predict_proba(X)[0][1])

        # (1) smooth the probability so it doesn't slam to 0/1
        p = _smooth_probability(raw_p, temperature=2.0, floor=0.02, ceil=0.98)

        prediction = _risk_from_prob(p, threshold=0.5)
        probability_pct = round(p * 100, 1)

        # save to DB
        row = Prediction(
            age=age, bmi=bmi, exercise_level=exercise_level,
            systolic_bp=systolic_bp, diastolic_bp=diastolic_bp, heart_rate=heart_rate,
            risk_label=prediction, probability=p
        )
        db.session.add(row)
        db.session.commit()

    return render_template("index.html", prediction=prediction, probability_pct=probability_pct, form=form)


# (2) Filter + (3) Pagination
@app.route("/history")
@login_required
def history():
    risk = request.args.get("risk", "All")   # All | High | Low
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 10))
    per_page = max(5, min(per_page, 50))

    q = Prediction.query

    if risk in ("High", "Low"):
        q = q.filter(Prediction.risk_label == risk)

    q = q.order_by(Prediction.created_at.desc())

    total = q.count()
    pages = max(1, math.ceil(total / per_page))
    page = max(1, min(page, pages))

    rows = q.offset((page - 1) * per_page).limit(per_page).all()

    return render_template(
        "history.html",
        rows=rows,
        risk=risk,
        page=page,
        pages=pages,
        per_page=per_page,
        total=total
    )


@app.route("/delete/<int:row_id>", methods=["POST"])
@login_required
def delete_row(row_id):
    row = Prediction.query.get_or_404(row_id)
    db.session.delete(row)
    db.session.commit()
    flash("Row deleted.")
    return redirect(url_for("history", risk=request.args.get("risk", "All")))


@app.route("/clear_history", methods=["POST"])
@login_required
def clear_history():
    Prediction.query.delete()
    db.session.commit()
    flash("History cleared.")
    return redirect(url_for("history"))


@app.route("/download_history_csv")
@login_required
def download_history_csv():
    rows = Prediction.query.order_by(Prediction.created_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["time_utc", "age", "bmi", "exercise", "systolic", "diastolic", "heart_rate", "risk", "probability"])

    for r in rows:
        writer.writerow([
            r.created_at.isoformat(),
            r.age, r.bmi, r.exercise_level,
            r.systolic_bp, r.diastolic_bp, r.heart_rate,
            r.risk_label, round((r.probability or 0) * 100, 1)
        ])

    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)

    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="predictions.csv")


# (4) Metrics Dashboard (simple + useful)
@app.route("/metrics")
@login_required
def metrics():
    # last 30 days counts by day
    by_day = (
        db.session.query(
            func.date(Prediction.created_at).label("day"),
            func.count(Prediction.id).label("count"),
            func.avg(Prediction.probability).label("avg_prob")
        )
        .group_by(func.date(Prediction.created_at))
        .order_by(func.date(Prediction.created_at).desc())
        .limit(30)
        .all()
    )

    # reverse to show oldest->newest on chart
    by_day = list(reversed(by_day))

    labels = [str(d.day) for d in by_day]
    counts = [int(d.count) for d in by_day]
    avg_probs = [round(float(d.avg_prob or 0) * 100, 1) for d in by_day]

    # totals
    total = Prediction.query.count()
    high = Prediction.query.filter(Prediction.risk_label == "High").count()
    low = Prediction.query.filter(Prediction.risk_label == "Low").count()

    return render_template(
        "metrics.html",
        labels=labels,
        counts=counts,
        avg_probs=avg_probs,
        total=total,
        high=high,
        low=low
    )
