import os
from flask import Flask

from .config import Config
from .extensions import db

def create_app():
    # Point Flask to the ROOT templates/static folders (not era/templates)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    templates_dir = os.path.join(base_dir, "templates")
    static_dir = os.path.join(base_dir, "static")

    app = Flask(__name__, template_folder=templates_dir, static_folder=static_dir)
    app.config.from_object(Config)

    # Init extensions
    db.init_app(app)

    # Import blueprints *inside* create_app to avoid import-time crashes
    # API blueprint
    try:
        from .api.routes import api_bp
    except Exception:
        # fallback names (in case you named it bp or blp)
        from .api.routes import bp as api_bp

    app.register_blueprint(api_bp, url_prefix="/api/v1")

    # WEB/UI blueprint (if you have root routes.py with web_bp)
    # If you don't have it, this safely skips.
    try:
        from routes import web_bp
        app.register_blueprint(web_bp)
    except Exception:
        pass

    with app.app_context():
        db.create_all()

    return app
