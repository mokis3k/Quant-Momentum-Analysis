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
        df = df[df['LineLength'] >= 150].copy()

        # target variable
        df['target'] = ((df['LineLength'] >= 150) & (df['LineLength'] <= 175)).astype(int)

        # start time as a separate feature
        df['Hour'] = pd.to_datetime(df['StartTime'], format="%H:%M").dt.hour

        return df

    def parse_duration(self, duration_str):
        try:
            parts = str(duration_str).split(':')
            if len(parts) == 2:  # "HH:MM"
                hours, minutes = parts
                seconds = 0
            elif len(parts) == 3:  # "HHH:MM:SS"
                hours, minutes, seconds = parts
            else:
                return None
            return float(hours) + float(minutes)/60 + float(seconds)/3600
        except:
            return None

    def prepare_features(self, df: pd.DataFrame):
        X = df[['Direction', 'Hour', 'Angle', 'Weekday', 'Duration']].copy()
        y = df['target']

        # Direction: BUY=1, SELL=0
        X['Direction'] = X['Direction'].map({'BUY': 1, 'SELL': 0})

        if pd.api.types.is_numeric_dtype(X['Duration']):
            X['Duration'] = X['Duration'].astype(float)
        else:
            X['Duration'] = X['Duration'].apply(self.parse_duration)

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
        df_new['Direction'] = df_new['Direction'].map({'BUY': 1, 'SELL': 0})

        if 'Duration' in df_new and df_new['Duration'].dtype == "object":
            df_new['Duration'] = df_new['Duration'].apply(self.parse_duration)

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
