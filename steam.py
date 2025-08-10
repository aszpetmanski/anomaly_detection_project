from flask import Flask, jsonify
import pandas as pd
import threading
import time

app = Flask(__name__)

# Load your dataset
df = pd.read_csv("streaming_dataset.csv")
current_index = 0
current_data = df.iloc[current_index].to_dict()

def stream_data():
    global current_index, current_data
    while True:
        current_data = df.iloc[current_index].to_dict()
        current_index = (current_index + 1) % len(df)  # loop back to 0
        time.sleep(1)

@app.route('/current', methods=['GET'])
def get_current_data():
    return jsonify(current_data)

@app.route('/start', methods=['GET'])
def start_stream():
    thread = threading.Thread(target=stream_data)
    thread.daemon = True
    thread.start()
    return "Streaming started.\n"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
