from era import create_app
from flask import render_template

app = create_app()

@app.route("/")
def home():
    return render_template("main.html")
