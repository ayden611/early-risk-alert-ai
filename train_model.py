import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib

# DEMO data (replace with real data later)
# Columns: age, bmi, exercise, sys_bp, dia_bp, smoker
X = np.array([
    [25, 22.0, 2, 118, 76, 0],
    [55, 33.0, 0, 150, 95, 1],
    [40, 28.0, 1, 135, 85, 0],
    [65, 36.0, 0, 160, 100, 1],
    [30, 24.0, 2, 120, 78, 0],
])

# Labels: 0 = Low Risk, 1 = High Risk
y = np.array([0, 1, 0, 1, 0])

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "demo_model.pkl")
print("Saved demo_model.pkl")
