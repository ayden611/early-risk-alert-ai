from flask import Flask
from .config import Config
from .extensions import db
from .api.routes import api_bp
from .web.routes import web_bp
from flask_login import LoginManager
import os

login_manager = LoginManager()


@login_manager.user_loader
def load_user(user_id):
    return None


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # -------------------------
    # Database Configuration
    # -------------------------
    db_url = os.getenv("SQLALCHEMY_DATABASE_URI") or os.getenv("DATABASE_URL")

    if db_url and db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    app.config["SQLALCHEMY_DATABASE_URI"] = db_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # -------------------------
    # Initialize Extensions
    # -------------------------
    login_manager.init_app(app)
    db.init_app(app)

    # -------------------------
    # Register Blueprints
    # -------------------------
    app.register_blueprint(api_bp)
    app.register_blueprint(web_bp)

    # -------------------------
    # Root Route
    # -------------------------
    @app.get("/")
    def root():
        return {"status": "running", "service": "early-risk-alert-ai"}

    # -------------------------
    # Health Check Route
    # -------------------------
    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    return app
