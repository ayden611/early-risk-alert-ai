import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

CSV_PATH = "data/health_data.csv"
MODEL_PATH = "demo_model.pkl"

def main():
    df = pd.read_csv(CSV_PATH)

    required = ["age","bmi","exercise","sys_bp","dia_bp","heart_rate","label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = df[["age","bmi","exercise","sys_bp","dia_bp","heart_rate"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model accuracy: {acc:.3f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")

if __name__ == "__main__":
    main()
