import pandas as pd
import shap
from model import ReversalModel

# Загружаем модель
model = ReversalModel()
model.load_model()

# Загружаем данные
df = pd.read_csv("./EURUSD/MLDataset.csv", sep=";", encoding="utf-8")
df['Hour'] = pd.to_datetime(df['StartTime'], format="%H:%M").dt.hour

X = df[['Direction', 'Hour', 'Angle', 'Weekday', 'Duration']].copy()
X['Direction'] = X['Direction'].map({'BUY': 1, 'SELL': 0})

# Duration в часы
def parse_duration(d):
    parts = str(d).split(':')
    if len(parts) == 2:
        hours, minutes = parts
        seconds = 0
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        return 0
    return float(hours) + float(minutes)/60 + float(seconds)/3600

X['Duration'] = X['Duration'].apply(parse_duration)
X = X[model.feature_columns]

# Новый API SHAP для RandomForest
explainer = shap.Explainer(model.model, X)  # background dataset = X
shap_values = explainer(X)                  # вернет объект с shap_values.values

# shap_values.values.shape => (n_samples, n_features, n_classes)
print("shap_values.values.shape:", shap_values.values.shape)

# Для класса 1 (разворот в диапазоне 150-175)
shap_values_class1 = shap_values.values[:, :, 1]

# Строим summary plot
shap.summary_plot(shap_values_class1, X)
