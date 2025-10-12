import socket
import json
import pickle
import traceback
import numpy as np

MODEL_PATH = "reversal_model.pkl"
HOST = "127.0.0.1"
PORT = 9999
BUFFER_SIZE = 4096

SL = 25
TP = 50
RISK = 0.01

def validate_data(data):
    required_keys = [
        "direction", "idx", "Start", "End",
        "LineLength", "AvgDiff", "ImpulseDiff",
        "PrevLineLength", "MaxDeviation"
    ]
    for key in required_keys:
        if key not in data:
            return False
    if data["direction"] not in ("BUY", "SELL"):
        return False
    return True

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict(model, features):
    X = np.array([[
        features["Start"],
        features["End"],
        features["LineLength"],
        features["AvgDiff"],
        features["ImpulseDiff"],
        features["PrevLineLength"],
        features["MaxDeviation"]
    ]])
    prob = float(model.predict_proba(X)[0][1])
    return prob

def reverse_action(direction):
    return "SELL" if direction == "BUY" else "BUY"

def main():
    model = load_model()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"Server started on {HOST}:{PORT}")

    while True:
        conn, addr = server.accept()
        try:
            data = conn.recv(BUFFER_SIZE).decode("utf-8").strip()
            if not data:
                continue

            try:
                request = json.loads(data)
            except json.JSONDecodeError:
                response = {"error": "Invalid JSON format"}
                conn.sendall(json.dumps(response).encode("utf-8"))
                continue

            if not validate_data(request):
                response = {"error": "Invalid data structure"}
                conn.sendall(json.dumps(response).encode("utf-8"))
                continue

            probability = predict(model, request)
            action = reverse_action(request["direction"])

            response = {
                "probability": round(probability, 4),
                "action": action,
                "sl": SL,
                "tp": TP,
                "risk": RISK
            }

            conn.sendall(json.dumps(response).encode("utf-8"))

        except Exception:
            err = traceback.format_exc()
            conn.sendall(json.dumps({"error": err}).encode("utf-8"))
        finally:
            conn.close()

if __name__ == "__main__":
    main()
