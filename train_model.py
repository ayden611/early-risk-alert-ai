import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = pd.read_csv("data/health_data.csv")

# Features (X) and Target (y)
X = data[["age", "bmi", "exercise", "sys_bp", "dia_bp", "heart_rate"]]
y = data["label"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save trained model
joblib.dump(model, "demo_model.pkl")

print("Model trained and saved as demo_model.pkl")
