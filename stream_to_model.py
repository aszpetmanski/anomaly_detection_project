import requests
import time
import json

STREAM_URL = "http://127.0.0.1:5000/current"
MLFLOW_URL = "http://127.0.0.1:5001/invocations"
HEADERS = {"Content-Type": "application/json"}

# Load features from features.json
with open("features.json", "r") as f:
    FEATURES = json.load(f)

def fetch_current_sensor_data():
    response = requests.get(STREAM_URL)
    return response.json()

def send_to_model(data):
    # Keep only model input fields dynamically
    model_input = {feature: data.get(feature, None) for feature in FEATURES}

    mlflow_payload = {
        "inputs": [model_input]
    }

    response = requests.post(MLFLOW_URL, headers=HEADERS, data=json.dumps(mlflow_payload))
    return response.json()

def run_loop():
    while True:
        try:
            sensor_data = fetch_current_sensor_data()
            prediction = send_to_model(sensor_data)
            print(f"\nSensor: {sensor_data}")
            print(f"Model Response: {prediction}")
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(1)

if __name__ == "__main__":
    print("Starting real-time model connector...\n")
    run_loop()








"""import requests
import time
import json

STREAM_URL = "http://127.0.0.1:5000/current"
MLFLOW_URL = "http://127.0.0.1:5001/invocations"
HEADERS = {"Content-Type": "application/json"}

def fetch_current_sensor_data():
    response = requests.get(STREAM_URL)
    return response.json()

def send_to_model(data):
    # Keep only model input fields
    model_input = {
        "temperature": data["temperature"],
        "humidity": data["humidity"],
        "sound_volume": data["sound_volume"]
    }

    mlflow_payload = {
        "inputs": [model_input]
    }

    response = requests.post(MLFLOW_URL, headers=HEADERS, data=json.dumps(mlflow_payload))
    return response.json()


def run_loop():
    while True:
        try:
            sensor_data = fetch_current_sensor_data()
            prediction = send_to_model(sensor_data)
            print(f"\nSensor: {sensor_data}")
            print(f"Model Response: {prediction}")
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(1)

if __name__ == "__main__":
    print("Starting real-time model connector...\n")
    run_loop()"""
