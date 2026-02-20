from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), "demo_model.pkl")
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    # Always define form_vals so template never crashes
    form_vals = {
        "age": "",
        "bmi": "",
        "exercise": "",
        "sys_bp": "",
        "dia_bp": "",
        "heart_rate": ""
    }

    if request.method == "POST":
        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        exercise = float(request.form["exercise"])
        sys_bp = float(request.form["sys_bp"])
        dia_bp = float(request.form["dia_bp"])
        heart_rate = float(request.form["heart_rate"])

        # Save values back to form
        form_vals = {
            "age": age,
            "bmi": bmi,
            "exercise": exercise,
            "sys_bp": sys_bp,
            "dia_bp": dia_bp,
            "heart_rate": heart_rate
        }

        data = np.array([[age, bmi, exercise, sys_bp, dia_bp, heart_rate]])

        pred = int(model.predict(data)[0])
        prob_high = float(model.predict_proba(data)[0][1])

        prediction = "High Risk" if pred == 1 else "Low Risk"
        probability = round(prob_high * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        form_vals=form_vals
    )


if __name__ == "__main__":
    app.run(debug=True)
