from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("demo_model.pkl")

@app.route("/")
def home():
    return "Early Risk Alert AI is Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    prediction = model.predict([data])
    return f"Prediction: {prediction[0]}"

if __name__ == "__main__":
    app.run(debug=True)
