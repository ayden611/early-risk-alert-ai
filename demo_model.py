# Early Risk Alert - Simple Naive Bayes Demo

import numpy as np
from sklearn.naive_bayes import GaussianNB

# Example dataset:
# [Age, BMI, Exercise Level]
# Exercise Level: 0 = Low, 1 = Medium, 2 = High

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

# 0 = Low Risk
# 1 = High Risk

y = np.array([0, 1, 1, 0, 1, 1, 0, 1])

# Create model
model = GaussianNB()

# Train model
model.fit(X, y)

# New patient example
new_patient = np.array([[50, 32, 0]])

prediction = model.predict(new_patient)
probability = model.predict_proba(new_patient)

print("Prediction (0 = Low Risk, 1 = High Risk):", prediction[0])
print("Probability:", probability)
