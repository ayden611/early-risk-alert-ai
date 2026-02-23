from flask import Flask, render_template, request, jsonify
from predict import predict_risk

app = Flask(__name__)

@app.get("/health")
def health():
    return jsonify(status="ok")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":
        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        exercise_level = request.form["exercise_level"]
        systolic_bp = float(request.form["systolic_bp"])
        diastolic_bp = float(request.form["diastolic_bp"])
        heart_rate = float(request.form["heart_rate"])

        label, prob_high = predict_risk(
            age=age,
            bmi=bmi,
            exercise_level=exercise_level,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            heart_rate=heart_rate,
        )

        prediction = label
        probability = round(prob_high * 100, 2)

    return render_template("index.html", prediction=prediction, probability=probability)
