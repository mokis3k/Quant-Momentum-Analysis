from model import ReversalModel

if __name__ == "__main__":
    model = ReversalModel()

    # train the model
    df = model.load_data("./MR-v1/MLDataset.csv")
    model.train(df)

    # test examples
    newLines = [
        {
            "Direction": "SELL",
            "Hour": 15,
            "Angle": 1.1,
            "Weekday": 2,
            "Duration": 24
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

    # obtain probabilities
    for i, line in enumerate(newLines, start=1):
        prob = model.predict_probability(line)
        print(f"[CASE {i}] Probability of reversal within the range 150-175: {prob:.2%}")
