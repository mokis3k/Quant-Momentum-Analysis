from model import ReversalModel

if __name__ == "__main__":
    model = ReversalModel()

    # 1. обучаем модель
    df = model.load_data("./MR-v1/MLDataset.csv")
    model.train(df)

    # 2. тестовые примеры (без LineLength!)
    newLines = [
        {
            "Direction": "SELL",
            "Hour": 15,
            "Angle": 1.1,
            "Weekday": 2,
            "Duration": 24  # часы, можно подставить реальные значения
        },
        {
            "Direction": "BUY",
            "Hour": 10,
            "Angle": 8.8,
            "Weekday": 3,
            "Duration": 12
        },
        {
            "Direction": "SELL",
            "Hour": 20,
            "Angle": 0.5,
            "Weekday": 4,
            "Duration": 36
        }
    ]

    # 3. получаем вероятности
    for i, line in enumerate(newLines, start=1):
        prob = model.predict_probability(line)
        print(f"[CASE {i}] Вероятность разворота в диапазоне 150-175: {prob:.2%}")
