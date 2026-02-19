import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from config import FEATURES, MODEL_FILE

def main():
    # Load dataset (adjust path if your CSV is elsewhere)
    df = pd.read_csv("data/health_data.csv")

    # Basic cleanup (drop rows missing required fields)
    df = df.dropna(subset=FEATURES + ["risk_label"])

    X = df[FEATURES].astype(float)
    y = df["risk_label"].astype(int)  # 0=Low, 1=High

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Save model to repo root
    joblib.dump(model, MODEL_FILE)

    print(f"Saved model -> {MODEL_FILE}")
    print(f"Test accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
