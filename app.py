from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("demo_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        exercise = float(request.form["exercise"])
        sys_bp = float(request.form["sys_bp"])
        dia_bp = float(request.form["dia_bp"])
        heart_rate = float(request.form["heart_rate"])

        features = np.array([[age, bmi, exercise, sys_bp, dia_bp, heart_rate]])

        result = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        prediction = "High Risk" if result == 1 else "Low Risk"
        probability = round(prob * 100, 2)

    return render_template("index.html", prediction=prediction, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
