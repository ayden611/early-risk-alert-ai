from flask import Flask
from .config import Config
from .extensions import db
from .api.routes import api_bp
from .web.routes import web_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)

    app.register_blueprint(api_bp, url_prefix="/api/v1")
    app.register_blueprint(web_bp)

    return app
