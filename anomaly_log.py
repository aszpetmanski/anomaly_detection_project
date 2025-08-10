import os, json, time, csv
from datetime import datetime
import requests

STREAM_URL = "http://127.0.0.1:5000/current"
MLFLOW_URL = "http://127.0.0.1:5001/invocations"
HEADERS    = {"Content-Type": "application/json"}
LOG_FILE   = "anomalies_log.csv"

# 1) load the exact features the model expects
with open("features.json", "r") as f:
    FEATURES = json.load(f)

def fetch_current():
    r = requests.get(STREAM_URL, timeout=2)
    r.raise_for_status()
    return r.json()

def to_mlflow_inputs(sensor):
    # keep only model features; ignore sim label or extras
    cleaned = {k: sensor.get(k, None) for k in FEATURES}
    return {"inputs": [cleaned]}, cleaned

def call_model(payload):
    r = requests.post(MLFLOW_URL, headers=HEADERS, data=json.dumps(payload), timeout=3)
    r.raise_for_status()
    return r.json()

def ensure_header(fieldnames):
    new_file = not os.path.exists(LOG_FILE)
    if new_file:
        with open(LOG_FILE, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

def log_anomaly_row(row_dict):
    # append a row dict to CSV
    with open(LOG_FILE, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=row_dict.keys()).writerow(row_dict)

def main():
    # dynamic header: timestamp + FEATURES + score + anomaly
    header = ["timestamp", *FEATURES, "anomaly_score", "anomaly"]
    ensure_header(header)

    print("Dynamic anomaly logger running (1 Hz). Ctrl+C to stop.")
    while True:
        try:
            sensor = fetch_current()
            payload, cleaned = to_mlflow_inputs(sensor)
            resp = call_model(payload)

            # mlflow pyfunc might return either {"predictions":[...]} or a list [...]
            pred = resp["predictions"][0] if isinstance(resp, dict) and "predictions" in resp else resp[0]
            anomaly = pred.get("anomaly")
            score   = pred.get("anomaly_score")

            # only log anomalies
            if anomaly == 1:
                row = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    **{k: cleaned.get(k) for k in FEATURES},
                    "anomaly_score": score,
                    "anomaly": anomaly,
                }
                log_anomaly_row(row)
                print(f"[ALERT] Logged anomaly: {row}")

            # optional: console trace
            print(f"Sensor: {cleaned}  ->  Pred: {pred}")

        except Exception as e:
            print(f"[warn] {e}")
            time.sleep(2)  # brief backoff
        time.sleep(1)

if __name__ == "__main__":
    main()