from flask import Flask, render_template, request
import numpy as np
import joblib
import os
import subprocess

app = Flask(__name__)

# Path to model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), "demo_model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        subprocess.run(["python3", "train_model.py"], check=True)
    return joblib.load(MODEL_PATH)

model = load_model()

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":
        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        sys_bp = float(request.form["sys_bp"])
        dia_bp = float(request.form["dia_bp"])
        heart_rate = float(request.form["heart_rate"])
        exercise = float(request.form["exercise"])

        # IMPORTANT: feature order must match training exactly
        data = np.array([[age, bmi, sys_bp, dia_bp, heart_rate, exercise]])

        prediction_value = int(model.predict(data)[0])
        prob = model.predict_proba(data)[0]

        probability = round(float(prob[1]) * 100, 2)  # class 1 = High Risk
        prediction = "High Risk" if prediction_value == 1 else "Low Risk"

    return render_template("index.html", prediction=prediction, probability=probability)
