from flask import Flask, request, render_template
import numpy as np
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

# Simple training data
X = np.array([
    [25, 22, 2],
    [45, 28, 0],
    [52, 31, 0],
    [23, 24, 2],
    [40, 30, 1],
    [60, 35, 0],
    [30, 26, 1],
    [55, 33, 0]
])

y = np.array([0, 1, 1, 0, 1, 1, 0, 1])

model = GaussianNB()
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None

    if request.method == "POST":
        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        exercise = float(request.form["exercise"])

        data = np.array([[age, bmi, exercise]])
        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][pred]

        prediction = "High Risk" if pred == 1 else "Low Risk"
        probability = round(float(prob), 2)

    return render_template("index.html",
                           prediction=prediction,
                           probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
