import os
import uuid
import datetime as dt
import numpy as np
import joblib

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    g,
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

# ----------------------------
# App Config
# ----------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret")

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL or "sqlite:///local.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ----------------------------
# Load Model
# ----------------------------
MODEL_PATH = "demo_model.pkl"
model = joblib.load(MODEL_PATH)

# ----------------------------
# Anonymous User System
# ----------------------------
CLIENT_ID_COOKIE = "client_id"
CLIENT_ID_HEADER = "X-Client-Id"


def get_or_create_client_id():
    header_id = request.headers.get(CLIENT_ID_HEADER)
    if header_id:
        return header_id

    cookie_id = request.cookies.get(CLIENT_ID_COOKIE)
    if cookie_id:
        return cookie_id

    return str(uuid.uuid4())


@app.before_request
def attach_client():
    g.client_id = get_or_create_client_id()


@app.after_request
def set_client_cookie(response):
    if not request.headers.get(CLIENT_ID_HEADER):
        if not request.cookies.get(CLIENT_ID_COOKIE):
            response.set_cookie(
                CLIENT_ID_COOKIE,
                g.client_id,
                max_age=60 * 60 * 24 * 365 * 2,
                httponly=True,
                secure=True,
                samesite="Lax",
            )
    return response


# ----------------------------
# Database Model
# ----------------------------
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    client_id = db.Column(db.String(64), index=True)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)

    age = db.Column(db.Integer)
    bmi = db.Column(db.Float)
    exercise_level = db.Column(db.String(20))
    systolic_bp = db.Column(db.Integer)
    diastolic_bp = db.Column(db.Integer)
    heart_rate = db.Column(db.Integer)

    risk_label = db.Column(db.String(20))
    probability = db.Column(db.Float)


with app.app_context():
    db.create_all()

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form["age"])
    bmi = float(request.form["bmi"])
    exercise = request.form["exercise_level"]
    systolic = int(request.form["systolic_bp"])
    diastolic = int(request.form["diastolic_bp"])
    heart_rate = int(request.form["heart_rate"])

    exercise_map = {"Low": 0, "Moderate": 1, "High": 2}
    exercise_val = exercise_map.get(exercise, 0)

    data = np.array([[age, bmi, exercise_val, systolic, diastolic, heart_rate]])

    pred = int(model.predict(data)[0])
    prob = float(model.predict_proba(data)[0][1])

    risk_label = "High Risk" if pred == 1 else "Low Risk"

    record = Prediction(
        client_id=g.client_id,
        age=age,
        bmi=bmi,
        exercise_level=exercise,
        systolic_bp=systolic,
        diastolic_bp=diastolic,
        heart_rate=heart_rate,
        risk_label=risk_label,
        probability=round(prob * 100, 2),
    )

    db.session.add(record)
    db.session.commit()

    return redirect(url_for("history"))


@app.route("/history")
def history():
    rows = (
        Prediction.query.filter_by(client_id=g.client_id)
        .order_by(Prediction.id.desc())
        .all()
    )
    return render_template("history.html", rows=rows)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json()

    age = payload["age"]
    bmi = payload["bmi"]
    exercise = payload["exercise_level"]
    systolic = payload["systolic_bp"]
    diastolic = payload["diastolic_bp"]
    heart_rate = payload["heart_rate"]

    exercise_map = {"Low": 0, "Moderate": 1, "High": 2}
    exercise_val = exercise_map.get(exercise, 0)

    data = np.array([[age, bmi, exercise_val, systolic, diastolic, heart_rate]])

    pred = int(model.predict(data)[0])
    prob = float(model.predict_proba(data)[0][1])
    risk_label = "High Risk" if pred == 1 else "Low Risk"

    record = Prediction(
        client_id=g.client_id,
        age=age,
        bmi=bmi,
        exercise_level=exercise,
        systolic_bp=systolic,
        diastolic_bp=diastolic,
        heart_rate=heart_rate,
        risk_label=risk_label,
        probability=round(prob * 100, 2),
    )

    db.session.add(record)
    db.session.commit()

    return jsonify(
        {
            "risk_label": risk_label,
            "probability": round(prob * 100, 2),
        }
    )


@app.route("/health")
def health():
    return {"status": "ok"}, 200


# ----------------------------
# Run (for local only)
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
