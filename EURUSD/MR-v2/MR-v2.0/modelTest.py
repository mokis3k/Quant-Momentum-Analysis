from model import ReversalModel

if __name__ == "__main__":
    model = ReversalModel()

    # train the model
    df = model.load_data("./EnrichedDataset.csv")
    model.train(df)

    # test examples
    newLines = [
        {
            "Direction": "SELL",
            "AvgDiff": 0.044,
            "ImpulseDiff": 0.1573,
            "PrevLineLength": -189.0625,
            "MaxDeviation": 0.6843,
            "Duration": 10695,
            "Angle": 1.03787,
            "Session": 2
        },
        {
            "Direction": "SELL",
            "AvgDiff": 0.0605,
            "ImpulseDiff": 0.6257,
            "PrevLineLength": -49.0196,
            "MaxDeviation": 0.6346,
            "Duration": 990,
            "Angle": 9.21212,
            "Session": 0
        },
        {
            "Direction": "BUY",
            "AvgDiff": 0.0489,
            "ImpulseDiff": 0.1028,
            "PrevLineLength": -163.8889,
            "MaxDeviation": 0.2926,
            "Duration": 5250,
            "Angle": 2.17143,
            "Session": 1
        },
        {
            "Direction": "SELL",
            "AvgDiff": 0.092,
            "ImpulseDiff": 0.6461,
            "PrevLineLength": -288.7097,
            "MaxDeviation": 0.7454,
            "Duration": 1755,
            "Angle": 8.23932,
            "Session": 1
        },
        {
            "Direction": "BUY",
            "AvgDiff": 0.0941,
            "ImpulseDiff": 0.5483,
            "PrevLineLength": -150.0,
            "MaxDeviation": 0.5314,
            "Duration": 1380,
            "Angle": 7.6087,
            "Session": 2
        }
    ]

    # obtain probabilities
    for i, line in enumerate(newLines, start=1):
        prob = model.predict_probability(line)
        print(f"[CASE {i}] Probability of reversal within the range 150-175: {prob:.2%}")
