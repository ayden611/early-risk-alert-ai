from flask import Blueprint, render_template_string
import era

web_bp = Blueprint("web", __name__)

MAIN_HTML = getattr(era, "MAIN_HTML", """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Early Risk Alert AI</title>
</head>
<body style="font-family:Arial,sans-serif;padding:40px;background:#07101c;color:#eef4ff;">
  <h1>Early Risk Alert AI</h1>
  <p>MAIN_HTML was not found.</p>
</body>
</html>
""")

# Single stable production entry:
# every public-facing route opens the working platform page
# so no investor/hospital/demo route can throw a 404/500.

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
    return render_template_string(MAIN_HTML)

@web_bp.get("/investor")
def investor_alias():
    return render_template_string(MAIN_HTML)

@web_bp.get("/investor-view")
def investor_view():
    return render_template_string(MAIN_HTML)

@web_bp.get("/hospital-demo")
def hospital_demo():
    return render_template_string(MAIN_HTML)

@web_bp.get("/hospital")
def hospital_alias():
    return render_template_string(MAIN_HTML)

@web_bp.get("/hospital-intake")
def hospital_intake():
    return render_template_string(MAIN_HTML)

@web_bp.get("/dashboard")
def dashboard():
    return render_template_string(MAIN_HTML)

@web_bp.get("/command-center")
def command_center():
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

@web_bp.get("/demo")
def demo():
    return render_template_string(MAIN_HTML)

@web_bp.get("/deck")
def deck():
    return render_template_string(MAIN_HTML)
