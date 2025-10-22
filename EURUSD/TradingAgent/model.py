import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ReversalModel:
    def __init__(self, model_path="reversal_model.pkl"):
        self.model_path = model_path
        self.pipeline = None
        self.feature_columns = None

    # -------------------- LOAD DATA --------------------
    def load_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, sep=";", encoding="utf-8")

        if {"StartTime", "EndTime"}.issubset(df.columns):
            df["StartTime"] = pd.to_datetime(df["StartTime"], format="%d.%m.%Y %H:%M", errors="coerce")
            df["EndTime"] = pd.to_datetime(df["EndTime"], format="%d.%m.%Y %H:%M", errors="coerce")
            if "Duration" not in df.columns:
                df["Duration"] = (df["EndTime"] - df["StartTime"]).dt.total_seconds() / 3600.0

        if "Angle" not in df.columns and {"LineLength", "Duration"}.issubset(df.columns):
            df["Angle"] = df["LineLength"] / df["Duration"].replace({0: pd.NA})

        df = df[df["LineLength"] >= 150].copy()
        df["target"] = ((df["LineLength"] >= 150) & (df["LineLength"] <= 175)).astype(int)

        return df

    # -------------------- PREPARE FEATURES --------------------
    def prepare_features(self, df: pd.DataFrame):
        feature_cols = [
            "AvgDiff",
            "LastCandlesDiff",
            "PrevLineLength",
            "MaxDeviation",
            "Duration",
            "Angle"
        ]

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

        X = df[feature_cols].copy()
        y = df["target"].astype(int)

        self.feature_columns = feature_cols
        return X, y

    # -------------------- TRAIN MODEL --------------------
    def train(self, df: pd.DataFrame):
        X, y = self.prepare_features(df)

        tscv = TimeSeriesSplit(n_splits=5)

        numeric_features = self.feature_columns
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features)])

        base_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            class_weight="balanced"
        )

        calibrated_model = CalibratedClassifierCV(base_model, cv=tscv, method="isotonic")

        self.pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", calibrated_model)
        ])

        self.pipeline.fit(X, y)

        # quality assessment on the last temporary split
        last_split_index = list(tscv.split(X))[-1]
        X_train_idx, X_test_idx = last_split_index
        X_test = X.iloc[X_test_idx]
        y_test = y.iloc[X_test_idx]
        probs = self.pipeline.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs) if len(set(y_test)) > 1 else float("nan")
        brier = brier_score_loss(y_test, probs)
        print(f"[INFO] AUC: {auc:.3f}, Brier score: {brier:.3f}")

        y_pred = (probs >= 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"[INFO] Accuracy: {accuracy:.3f}")
        print(f"[INFO] Precision: {precision:.3f}")
        print(f"[INFO] Recall: {recall:.3f}")
        print(f"[INFO] F1-score: {f1:.3f}")

        joblib.dump(self.pipeline, self.model_path)
        print(f"[INFO] Full pipeline saved to {self.model_path}")

    # -------------------- LOAD MODEL --------------------
    def load_model(self):
        self.pipeline = joblib.load(self.model_path)
        self.feature_columns = self.pipeline.named_steps["preprocessor"].transformers_[0][2]

    # -------------------- PREDICT PROBABILITY --------------------
    def predict_probability(self, new_data: dict) -> float:
        if self.pipeline is None:
            self.load_model()

        df_new = pd.DataFrame([new_data])

        if {"StartTime", "EndTime"}.issubset(df_new.columns) and "Duration" not in df_new.columns:
            df_new["StartTime"] = pd.to_datetime(df_new["StartTime"], format="%d.%m.%Y %H:%M", errors="coerce")
            df_new["EndTime"] = pd.to_datetime(df_new["EndTime"], format="%d.%m.%Y %H:%M", errors="coerce")
            df_new["Duration"] = (df_new["EndTime"] - df_new["StartTime"]).dt.total_seconds() / 3600.0

        if "Angle" not in df_new.columns and {"LineLength", "Duration"}.issubset(df_new.columns):
            df_new["Angle"] = df_new["LineLength"] / df_new["Duration"].replace({0: pd.NA})

        X = df_new.reindex(columns=self.feature_columns)
        if X.isnull().any(axis=None):
            raise ValueError("Missing or invalid numeric values in input features.")

        prob = self.pipeline.predict_proba(X)[:, 1][0]
        return float(prob)
