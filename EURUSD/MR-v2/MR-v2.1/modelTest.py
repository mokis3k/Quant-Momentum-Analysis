# modelTest.py
import json
from model import ReversalModel

if __name__ == "__main__":
    # model initialisation
    model = ReversalModel(model_path="reversal_model.pkl")

    # input data examples
    examples = [
        {
            "Direction": "SELL",
            "StartTime": "21.09.2021 16:30",
            "EndTime": "01.10.2021 04:00",
            "StartPrice": 1.17307,
            "EndPrice": 1.15635,
            "LineLength": 167,
            "AvgDiff": 0.0448,
            "LastCandlesDiff": 0.0315,
            "PrevLineLength": 0,
            "MaxDeviation": 0.8135,
            "Duration": 227.5,
            "Angle": 0.7341
        },
        {
            "Direction": "BUY",
            "StartTime": "11.10.2021 09:30",
            "EndTime": "12.10.2021 20:45",
            "StartPrice": 1.15861,
            "EndPrice": 1.15257,
            "LineLength": 60,
            "AvgDiff": 0.0401,
            "LastCandlesDiff": 0.0472,
            "PrevLineLength": -11.1111,
            "MaxDeviation": 0.2308,
            "Duration": 35.25,
            "Angle": 1.7021
        },
        {
            "Direction": "BUY",
            "StartTime": "13.01.2025 11:15",
            "EndTime": "15.01.2025 15:45",
            "StartPrice": 1.01868,
            "EndPrice": 1.03444,
            "LineLength": 158,
            "AvgDiff": 0.0755,
            "LastCandlesDiff": 0.1254,
            "PrevLineLength": 34.4398,
            "MaxDeviation": 0.4651,
            "Duration": 52.5,
            "Angle": 3.0095
        },
        {
            "Direction": "BUY",
            "StartTime": "20.01.2025 01:15",
            "EndTime": "20.01.2025 17:00",
            "StartPrice": 1.02683,
            "EndPrice": 1.04255,
            "LineLength": 158,
            "AvgDiff": 0.0876,
            "LastCandlesDiff": 0.2419,
            "PrevLineLength": -185.4545,
            "MaxDeviation": 0.9868,
            "Duration": 15.75,
            "Angle": 9.9683
        }
    ]

    # results
    for i, sample in enumerate(examples, 1):
        try:
            prob = model.predict_probability(sample)
            print(f"[TEST {i}] Input: {json.dumps(sample, ensure_ascii=False)}")
            print(f"→ Predicted reversal probability: {prob:.4f}\n")
        except Exception as e:
            print(f"[TEST {i}] Error: {e}\n")
