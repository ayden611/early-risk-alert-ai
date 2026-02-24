from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import os
from datetime import datetime

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    age = db.Column(db.Float)
    bmi = db.Column(db.Float)
    exercise_level = db.Column(db.Float)
    systolic_bp = db.Column(db.Float)
    diastolic_bp = db.Column(db.Float)
    heart_rate = db.Column(db.Float)
    risk_label = db.Column(db.String(50))
    probability = db.Column(db.Float)

# 🔥 FORCE RESET TABLE
with app.app_context():
    db.drop_all()
    db.create_all()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            age = float(request.form["age"])
            bmi = float(request.form["bmi"])
            exercise_level = float(request.form["exercise_level"])
            systolic_bp = float(request.form["systolic_bp"])
            diastolic_bp = float(request.form["diastolic_bp"])
            heart_rate = float(request.form["heart_rate"])

            probability = min(100, (bmi + systolic_bp + diastolic_bp) / 3)
            risk = "High Risk" if probability > 50 else "Low Risk"

            prediction = Prediction(
                age=age,
                bmi=bmi,
                exercise_level=exercise_level,
                systolic_bp=systolic_bp,
                diastolic_bp=diastolic_bp,
                heart_rate=heart_rate,
                risk_label=risk,
                probability=probability
            )

            db.session.add(prediction)
            db.session.commit()

            return render_template("index.html", result=risk, probability=round(probability, 2))

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

@app.route("/history")
def history():
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
    return render_template("history.html", predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)
