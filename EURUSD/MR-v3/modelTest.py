from model import ReversalModel

if __name__ == "__main__":
    model = ReversalModel()

    # train the model
    df = model.load_data("./EnrichedDataset.csv")
    model.train(df)

    # test examples
    newLines = [
        {
            "AvgDiff": 0.048,
            "PrevLineLength": -210.5,
            "MaxDeviation": 0.602,
            "MaxImpulse": 0.185,
            "Duration": 8420,
            "Angle": 1.52
        },
        {
            "AvgDiff": 0.056,
            "PrevLineLength": -125.7,
            "MaxDeviation": 0.355,
            "MaxImpulse": 0.612,
            "Duration": 6735,
            "Angle": 2.04
        },
        {
            "AvgDiff": 0.078,
            "PrevLineLength": -295.2,
            "MaxDeviation": 0.845,
            "MaxImpulse": 0.423,
            "Duration": 3920,
            "Angle": 3.27
        },
        {
            "AvgDiff": 0.067,
            "PrevLineLength": -80.9,
            "MaxDeviation": 0.278,
            "MaxImpulse": 0.712,
            "Duration": 5280,
            "Angle": 1.96
        },
        {
            "AvgDiff": 0.091,
            "PrevLineLength": -335.8,
            "MaxDeviation": 0.763,
            "MaxImpulse": 0.501,
            "Duration": 7125,
            "Angle": 2.81
        }
    ]

    # obtain probabilities
    for i, line in enumerate(newLines, start=1):
        prob = model.predict_probability(line)
        print(f"[CASE {i}] Probability of reversal within the range 150-175: {prob:.2%}")
