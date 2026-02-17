from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model
model = joblib.load("demo_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":
        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        exercise = float(request.form["exercise"])

        data = np.array([[age, bmi, exercise]])
        prediction = model.predict(data)[0]
        probability = model.predict_proba(data)[0].tolist()

    return render_template("index.html", prediction=prediction, probability=probability)

    if __name__ == "__main__":
    app.run(debug=True)
