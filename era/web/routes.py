from flask import Blueprint, render_template, send_file
import os

web_bp = Blueprint("web", __name__)


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _templates_dir():
    return os.path.join(_project_root(), "templates")


def _template_exists(filename):
    return os.path.exists(os.path.join(_templates_dir(), filename))


@web_bp.route("/")
def home():
    return render_template("command_center.html")


@web_bp.route("/investors")
def investors():
    return render_template("investor.html")


@web_bp.route("/login")
def login():
    if _template_exists("login.html"):
        return render_template("login.html")
    return render_template("investor.html")


@web_bp.route("/dashboard")
def dashboard():
    if _template_exists("dashboard.html"):
        return render_template("dashboard.html")
    if _template_exists("command_center.html"):
        return render_template("command_center.html")
    return render_template("investor.html")


@web_bp.route("/deck")
@web_bp.route("/pitch-deck")
def pitch_deck():
    pdf_path = os.path.join(_project_root(), "static", "Early_Risk_Alert_AI_Pitch_Deck.pdf")

    if not os.path.exists(pdf_path):
        return "Pitch deck not found.", 404

    return send_file(
        pdf_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="Early_Risk_Alert_AI_Pitch_Deck.pdf",
    )
