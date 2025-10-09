import pandas as pd
import shap
from model import ReversalModel

# Loading model
model = ReversalModel()
model.load_model()

# Loading data
df = pd.read_csv("./EnrichedDataset.csv", sep=";", encoding="utf-8")

# Filtering, as in the model
df = df[df["LineLength"] >= 150].copy()
df["target"] = ((df["LineLength"] >= 150) & (df["LineLength"] <= 175)).astype(int)

# Preparation of features
X = df[[
    "Direction",
    "AvgDiff",
    "ImpulseDiff",
    "PrevLineLength",
    "MaxDeviation",
    "Duration",
    "Angle",
    "Session"
]].copy()

# Encode Direction
X["Direction"] = X["Direction"].map({"BUY": 1, "SELL": 0})

# Convert all attributes to numeric type
numeric_cols = [
    "AvgDiff",
    "ImpulseDiff",
    "PrevLineLength",
    "MaxDeviation",
    "Duration",
    "Angle",
    "Session"
]
X[numeric_cols] = X[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Checking the order of columns
X = X[model.feature_columns]

# --- SHAP visualisation ---
# Creating an explainer based on the RandomForest model
explainer = shap.Explainer(model.model, X)

# Obtain shap values for all rows
shap_values = explainer(X)

print("shap_values.values.shape:", shap_values.values.shape)

# For class 1 (reverse in the range of 150–175)
shap_values_class1 = shap_values.values[:, :, 1]

# Summary plot
shap.summary_plot(shap_values_class1, X, show=True)
