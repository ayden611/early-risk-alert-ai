import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

# Columns (MUST match Flask order exactly):
# age, bmi, sys_bp, dia_bp, heart_rate, exercise

X = np.array([
    [25, 22.0, 118, 76, 72, 2],
    [55, 33.0, 150, 95, 88, 0],
    [40, 28.0, 135, 85, 76, 1],
    [65, 36.0, 160, 100, 92, 0],
    [30, 24.0, 120, 78, 70, 2],
])

# Labels: 0 = Low Risk, 1 = High Risk
y = np.array([0, 1, 0, 1, 0])

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "demo_model.pkl")
print("Saved demo_model.pkl")
