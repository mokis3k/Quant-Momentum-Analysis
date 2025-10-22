# ml_server_http.py
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import traceback
import joblib
import os

HOST = "127.0.0.1"
PORT = 5000
MODEL_PATH = "reversal_model.pkl"
DATA_PATH = "dataset.csv"
P_THRESHOLD = 0.8
SL = 25
TP = 50
RISK = 0.01

MODEL_WRAPPER = None
try:
    from model import ReversalModel
    rm = ReversalModel(MODEL_PATH)

    # First, try loading the existing model.
    if os.path.exists(MODEL_PATH):
        rm.load_model()
        print("[INFO] The model was loaded from the file.")
    else:
        # If the file does not exist, train a new one and save it
        if os.path.exists(DATA_PATH):
            df = rm.load_data(DATA_PATH)
            rm.train(df)
            print("[INFO] New model trained and saved.")
        else:
            raise FileNotFoundError("Model file and training dataset not found.")

    MODEL_WRAPPER = rm
except Exception as e:
    print(f"[WARN] Failed to initialise ReversalModel: {e}")
    try:
        MODEL_WRAPPER = joblib.load(MODEL_PATH)
        print("[INFO] Model loaded directly from pickle.")
    except Exception as ex:
        print("[ERROR] Model loading error:", ex)
        MODEL_WRAPPER = None


def build_features_from_payload(payload):
    feat = {
        "Direction": payload.get("direction", payload.get("Direction", "")).upper(),
        "AvgDiff": float(payload.get("AvgDiff", 0.0)),
        "ImpulseDiff": float(payload.get("ImpulseDiff", 0.0)),
        "PrevLineLength": float(payload.get("PrevLineLength", 0.0)),
        "MaxDeviation": float(payload.get("MaxDeviation", 0.0)),
        "Duration": float(payload.get("Duration", 0.0)),
        "Angle": float(payload.get("Angle", 0.0)),
        "LineLength": float(payload.get("LineLength", 0.0)),
    }
    return feat


def predict_probability(feat):
    if MODEL_WRAPPER is None:
        return 0.0
    try:
        if hasattr(MODEL_WRAPPER, "predict_probability"):
            return float(MODEL_WRAPPER.predict_probability(feat))
        elif isinstance(MODEL_WRAPPER, tuple) and len(MODEL_WRAPPER) == 2:
            model, cols = MODEL_WRAPPER
            import numpy as np
            row = [
                feat.get(c, 0.0) if c != "Direction" else (1 if feat.get("Direction", "BUY") == "BUY" else 0)
                for c in cols
            ]
            return float(model.predict_proba([row])[0, 1])
        else:
            import numpy as np
            model = MODEL_WRAPPER
            X = np.array([[feat["LineLength"], feat["AvgDiff"], feat["ImpulseDiff"],
                           feat["PrevLineLength"], feat["MaxDeviation"],
                           feat["Duration"], feat["Angle"]]])
            return float(model.predict_proba(X)[0, 1])
    except Exception:
        traceback.print_exc()
        return 0.0


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/line":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error":"invalid_json"}')
            return

        # Minimum validation
        needed = ["direction", "LineLength"]
        for k in needed:
            if k not in payload:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": "missing_field", "field": k}).encode())
                return

        feat = build_features_from_payload(payload)
        prob = predict_probability(feat)
        action = "NONE"
        if 150 <= feat["LineLength"] <= 175 and prob >= P_THRESHOLD:
            action = "SELL" if feat["Direction"] == "BUY" else "BUY"

        resp = {"probability": round(prob, 4), "action": action, "sl": SL, "tp": TP, "risk": RISK}
        resp_bytes = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_bytes)))
        self.end_headers()
        self.wfile.write(resp_bytes)


if __name__ == "__main__":
    httpd = HTTPServer((HOST, PORT), Handler)
    print(f"[INFO] HTTP ML server launched: http://{HOST}:{PORT}/api/line")
    httpd.serve_forever()
