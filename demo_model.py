# Early Risk Alert - Simple Naive Bayes Demo

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import joblib

# Feature matrix: [Age, BMI, Exercise Level]
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

# 0 = Low Risk, 1 = High Risk
y = np.array([0, 1, 1, 0, 1, 1, 0, 1])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create model
model = GaussianNB()

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Example new patient
new_patient = np.array([[50, 32, 0]])
prediction = model.predict(new_patient)
probability = model.predict_proba(new_patient)

print("\nNew Patient Prediction (0=Low Risk, 1=High Risk):", prediction[0])
print("Probability:", probability)

# saved trained model
joblib.dump(model, "demo_model.pkl")
print("model saved as demo_model.pkl")

