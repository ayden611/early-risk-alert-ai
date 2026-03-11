from flask import Blueprint, render_template_string
import era

web_bp = Blueprint("web", __name__)

MAIN_HTML = getattr(era, "MAIN_HTML", """
<!doctype html>
<html>
<head><title>Early Risk Alert AI</title></head>
<body style="font-family:Arial;padding:40px;background:#07101c;color:white;">
  <h1>Early Risk Alert AI</h1>
  <p>Main page is loading, but MAIN_HTML was not found.</p>
</body>
</html>
""")

INVESTOR_HTML = getattr(era, "INVESTOR_HTML", MAIN_HTML)
HOSPITAL_HTML = getattr(era, "HOSPITAL_HTML", MAIN_HTML)
COMMAND_CENTER_HTML = getattr(era, "COMMAND_CENTER_HTML", MAIN_HTML)


@web_bp.get("/")
def home():
    return render_template_string(MAIN_HTML)


@web_bp.get("/overview")
def overview():
    return render_template_string(MAIN_HTML)


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


@web_bp.get("/dashboard")
def dashboard():
    return render_template_string(COMMAND_CENTER_HTML)


@web_bp.get("/command-center")
def command_center():
    return render_template_string(COMMAND_CENTER_HTML)


@web_bp.get("/admin")
def admin():
    return render_template_string(MAIN_HTML)


@web_bp.get("/admin/review")
def admin_review():
    return render_template_string(MAIN_HTML)


@web_bp.get("/login")
def login():
    return render_template_string(MAIN_HTML)


@web_bp.get("/demo")
def demo():
    return render_template_string(MAIN_HTML)


@web_bp.get("/deck")
def deck():
    return render_template_string(MAIN_HTML)
