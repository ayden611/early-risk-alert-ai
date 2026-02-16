from flask import Flask, request, render_template_string
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

HTML = """
<h2>Early Risk Alert AI</h2>
<form method="post">
Age: <input type="number" name="age"><br><br>
BMI: <input type="number" name="bmi"><br><br>
Exercise Level (0=Low,1=Medium,2=High):
<input type="number" name="exercise"><br><br>
<input type="submit">
</form>

{% if prediction is not none %}
<h3>Prediction: {{ prediction }}</h3>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        exercise = float(request.form["exercise"])

        new_data = np.array([[age, bmi, exercise]])
        result = model.predict(new_data)

        prediction = "High Risk" if result[0] == 1 else "Low Risk"

    return render_template_string(HTML, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
