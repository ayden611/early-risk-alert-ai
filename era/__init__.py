import os
from flask import Flask

from .config import Config
from .extensions import db
from .api.routes import api_bp


def create_app():
    # Point Flask to ROOT templates/static folders (not era/templates)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    templates_dir = os.path.join(base_dir, "templates")
    static_dir = os.path.join(base_dir, "static")

    app = Flask(__name__, template_folder=templates_dir, static_folder=static_dir)
    app.config.from_object(Config)

    db.init_app(app)

    # Register API
    app.register_blueprint(api_bp, url_prefix="/api/v1")

    # Create DB tables
    with app.app_context():
        db.create_all()

    return app
