import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss


class ReversalModel:
    def __init__(self, model_path="reversal_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_columns = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        df = pd.read_csv(filepath, sep=";", encoding="utf-8")

        # filter short lines
        df = df[df["LineLength"] >= 150].copy()

        # target variable
        df["target"] = ((df["LineLength"] >= 150) & (df["LineLength"] <= 175)).astype(int)

        return df

    def prepare_features(self, df: pd.DataFrame):

        feature_cols = [
            "AvgDiff",
            "PrevLineLength",
            "MaxDeviation",
            "MaxImpulse",
            "Duration",
            "Angle"
        ]

        # all features are in the data
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Отсутствуют необходимые признаки: {missing}")

        X = df[feature_cols].copy()
        y = df["target"]

        # convert types
        numeric_cols = [
            "AvgDiff",
            "PrevLineLength",
            "MaxDeviation",
            "MaxImpulse",
            "Duration",
            "Angle"
        ]
        X[numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors="coerce")

        self.feature_columns = X.columns.tolist()
        return X, y

    def train(self, df: pd.DataFrame):
        X, y = self.prepare_features(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            class_weight="balanced"
        )
        self.model.fit(X_train, y_train)

        # quality assessment
        probs = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)

        print(f"[INFO] AUC: {auc:.3f}, Brier score: {brier:.3f}")

        # save model
        joblib.dump((self.model, self.feature_columns), self.model_path)
        print(f"[INFO] Model saved in {self.model_path}")

        # significance of signs
        self.feature_importance()

    def load_model(self):
        self.model, self.feature_columns = joblib.load(self.model_path)

    def predict_probability(self, new_data: dict) -> float:
        if self.model is None:
            self.load_model()

        df_new = pd.DataFrame([new_data])

        # the order of the columns matches
        df_new = df_new[self.feature_columns]

        prob = self.model.predict_proba(df_new)[:, 1][0]
        return prob

    def feature_importance(self):
        if self.model is None:
            print("[WARN] The model is not trained")
            return

        importances = self.model.feature_importances_
        for col, imp in sorted(zip(self.feature_columns, importances), key=lambda x: x[1], reverse=True):
            print(f"[INFO] Feature {col}: significance {imp:.4f}")
