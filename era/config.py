import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")

    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///predictions.db")
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # JWT for mobile/API later
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", SECRET_KEY)

    # Admin login (web)
    ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "password")

    # Probability smoothing (no retrain needed)
    PROB_TEMP = float(os.environ.get("PROB_TEMP", "2.0"))
    PROB_FLOOR = float(os.environ.get("PROB_FLOOR", "0.02"))
    PROB_CEIL = float(os.environ.get("PROB_CEIL", "0.98"))
