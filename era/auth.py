from flask_login import LoginManager

login_manager = LoginManager()
login_manager.login_view = "web.login"  # adjust if your login route name differs
