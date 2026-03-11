from flask import Blueprint, render_template_string

web_bp = Blueprint("web", __name__)

# These HTML strings must already exist in your project.
# MAIN_HTML should be your full working homepage/platform page.
# INVESTOR_HTML should be your investor page HTML.
# If you already have HOSPITAL_HTML or COMMAND_CENTER_HTML, use them below.
# If not, the routes safely fall back to MAIN_HTML so nothing breaks.

from era.__init__ import MAIN_HTML, INVESTOR_HTML

try:
    from era.__init__ import HOSPITAL_HTML
except Exception:
    HOSPITAL_HTML = MAIN_HTML

try:
    from era.__init__ import COMMAND_CENTER_HTML
except Exception:
    COMMAND_CENTER_HTML = MAIN_HTML


@web_bp.get("/")
def home():
    return render_template_string(MAIN_HTML)


@web_bp.get("/overview")
def overview():
    return render_template_string(MAIN_HTML)


@web_bp.get("/dashboard")
def dashboard():
    return render_template_string(COMMAND_CENTER_HTML)


@web_bp.get("/command-center")
def command_center():
    return render_template_string(COMMAND_CENTER_HTML)


@web_bp.get("/simulator")
def simulator():
    return render_template_string(MAIN_HTML)


@web_bp.get("/investors")
def investors():
    return render_template_string(INVESTOR_HTML)


@web_bp.get("/investor")
def investor_alias():
    return render_template_string(INVESTOR_HTML)


@web_bp.get("/investor-view")
def investor_view():
    return render_template_string(INVESTOR_HTML)


@web_bp.get("/hospital-demo")
def hospital_demo():
    return render_template_string(HOSPITAL_HTML)


@web_bp.get("/hospital")
def hospital_alias():
    return render_template_string(HOSPITAL_HTML)


@web_bp.get("/hospital-intake")
def hospital_intake():
    return render_template_string(HOSPITAL_HTML)


@web_bp.get("/demo")
def demo():
    return render_template_string(MAIN_HTML)


@web_bp.get("/admin")
def admin():
    return render_template_string(MAIN_HTML)


@web_bp.get("/admin/review")
def admin_review():
    return render_template_string(MAIN_HTML)


@web_bp.get("/login")
def login():
    return render_template_string(MAIN_HTML)


@web_bp.get("/deck")
def deck():
    return render_template_string(MAIN_HTML)
