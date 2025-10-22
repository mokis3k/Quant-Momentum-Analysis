import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

MODEL_PATH = "reversal_model_multi.pkl"
DATA_PATH = "dataset.csv"

def main():
    obj = joblib.load(MODEL_PATH)
    pipeline = obj["pipeline"]
    feature_columns = obj["feature_columns"]

    df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8")

    length_bins = list(range(50, 525, 25))
    labels = [f"{a}-{b}" for a, b in zip(length_bins[:-1], length_bins[1:])]
    df["target"] = pd.cut(df["LineLength"], bins=length_bins, labels=labels, include_lowest=True, right=False)
    df = df[df["target"].notna()].copy()

    X = df[feature_columns].copy()
    y_true = df["target"].astype(str)

    y_pred = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    acc = accuracy_score(y_true, y_pred)
    print(f"[TEST] Accuracy: {acc:.3f}")
    print("[TEST] Classification report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("[TEST] Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    sample_probs = dict(zip(pipeline.classes_, probs[0]))
    print("\n[TEST] Probabilities for first sample:")
    for k, v in sample_probs.items():
        print(f"  {k}: {v:.3f}")

if __name__ == "__main__":
    main()
