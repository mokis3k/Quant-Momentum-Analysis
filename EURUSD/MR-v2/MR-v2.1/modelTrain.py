from model import ReversalModel

model = ReversalModel()
df = model.load_data("GroupedLogs.csv")
model.train(df)