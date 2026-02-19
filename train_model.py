import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

# Columns: age, bmi, exercise, sys_bp, dia_bp, heart_rate
X = np.array([
    [25, 22.0, 2, 118, 76, 72],
    [55, 33.0, 0, 150, 95, 92],
    [40, 28.0, 1, 135, 85, 80],
    [65, 36.0, 0, 160, 100, 98],
    [30, 24.0, 2, 120, 78, 70],
])

# Labels: 0 = Low Risk, 1 = High Risk
y = np.array([0, 1, 0, 1, 0])

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "demo_model.pkl")
print("Saved demo_model.pkl")
