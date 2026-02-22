from flask import Flask
from .config import Config
from .extensions import db

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(Config)

    db.init_app(app)

    # API
    from .api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix="/api/v1")

    # WEB (homepage + history pages)
    from .web.routes import web_bp
    app.register_blueprint(web_bp)

    with app.app_context():
        db.create_all()

    return app
