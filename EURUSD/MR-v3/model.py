import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ReversalModelMulti:
    def __init__(self, model_path="reversal_model.pkl"):
        self.model_path = model_path
        self.pipeline = None
        self.feature_columns = [
            "LineLength",
            "AvgDiff",
            "LastCandlesDiff",
            "PrevLineLength",
            "MaxDeviation",
            "DurationHours",
            "BarsCount"
        ]
        self.length_bins = list(range(50, 525, 25))
        self.labels = [f"{a}-{b}" for a, b in zip(self.length_bins[:-1], self.length_bins[1:])]

    # -------------------- LOAD DATA --------------------
    def load_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, sep=";", encoding="utf-8")

        # формирование категориальной цели по диапазонам
        df["target"] = pd.cut(
            df["LineLength"],
            bins=self.length_bins,
            labels=self.labels,
            include_lowest=True,
            right=False
        )

        df = df[df["target"].notna()].copy()
        return df

    # -------------------- PREPARE FEATURES --------------------
    def prepare_features(self, df: pd.DataFrame):
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        X = df[self.feature_columns].copy()
        y = df["target"].astype(str)
        return X, y

    # -------------------- TRAIN MODEL --------------------
    def train(self, df: pd.DataFrame):
        X, y = self.prepare_features(df)

        tscv = TimeSeriesSplit(n_splits=5)

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, self.feature_columns)]
        )

        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            random_state=42,
            class_weight="balanced"
        )

        self.pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        self.pipeline.fit(X, y)

        # последняя проверка на последнем временном сплите
        train_idx, test_idx = list(tscv.split(X))[-1]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        y_pred = self.pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        print(f"[INFO] Accuracy: {acc:.3f}")
        print("[INFO] Classification report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("[INFO] Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        joblib.dump({
            "pipeline": self.pipeline,
            "feature_columns": self.feature_columns,
            "labels": self.labels
        }, self.model_path, compress=3)
        print(f"[INFO] Saved model to {self.model_path}")

    # -------------------- LOAD MODEL --------------------
    def load_model(self):
        obj = joblib.load(self.model_path)
        self.pipeline = obj["pipeline"]
        self.feature_columns = obj["feature_columns"]
        self.labels = obj["labels"]

    # -------------------- PREDICT PROBABILITY --------------------
    def predict_probability(self, new_data: dict) -> dict:
        if self.pipeline is None:
            self.load_model()

        df_new = pd.DataFrame([new_data])
        missing = [c for c in self.feature_columns if c not in df_new.columns]
        if missing:
            raise ValueError(f"Missing input features: {missing}")

        X = df_new.reindex(columns=self.feature_columns)
        probs = self.pipeline.predict_proba(X)[0]
        result = dict(zip(self.pipeline.classes_, probs))
        return result
