from datetime import datetime
from flask_login import UserMixin
from .extensions import db


# -----------------------------
# User Model
# -----------------------------
class User(db.Model, UserMixin):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


# -----------------------------
# Prediction Model
# -----------------------------
class Prediction(db.Model):
    __tablename__ = "predictions_v2"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    exercise_level = db.Column(db.String(20), nullable=False)

    systolic_bp = db.Column(db.Float, nullable=False)
    diastolic_bp = db.Column(db.Float, nullable=False)
    heart_rate = db.Column(db.Float, nullable=False)

    risk_label = db.Column(db.String(30), nullable=False)
    probability = db.Column(db.Float, nullable=False)


# -----------------------------
# Health Event Pipeline
# -----------------------------
class HealthEvent(db.Model):
    __tablename__ = "health_event"
    __table_args__ = {"extend_existing": True}

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.String(64), index=True)

    event_type = db.Column(db.String(50))

    data = db.Column(db.JSON)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
