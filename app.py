from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "demo_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":
        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        exercise = float(request.form["exercise"])
        sys_bp = float(request.form["sys_bp"])
        dia_bp = float(request.form["dia_bp"])
        heart_rate = float(request.form["heart_rate"])

        # IMPORTANT: order must match training
        data = np.array([[age, bmi, exercise, sys_bp, dia_bp, heart_rate]])

        prediction_value = int(model.predict(data)[0])
        prediction = "High Risk" if prediction_value == 1 else "Low Risk"

        prob = model.predict_proba(data)[0]
        probability = round(float(max(prob)) * 100, 2)

    return render_template("index.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
