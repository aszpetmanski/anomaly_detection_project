IoT Anomaly Detection – Isolation Forest & MLflow

📌 Overview

This project simulates real-time anomaly detection in a manufacturing environment using IoT sensor data.
We use a Flask application to stream sensor readings, an Isolation Forest model for detecting anomalies, and MLflow to package and serve the model as a REST API.
The system is designed to be dynamic, meaning it automatically adapts to new datasets with different features after retraining.


🚀 Features\
Synthetic IoT Data Simulation – Gaussian-distributed data with injected anomalies.\
Real-time Streaming – Flask endpoint sends one reading per second.\
Unsupervised Anomaly Detection – Isolation Forest, no labels required.\
Dynamic Feature Handling – Automatically adapts to new sensors.\
Model Serving via MLflow – REST API for predictions.\
Anomaly Logging – CSV file with timestamps, sensor readings, and scores.\
Retraining Pipeline – Easy to adapt to new factories or conditions.\


📂 Running the Project\
This repository contains two complete execution flows:\
Run with Dataset v1 (Original) –\
Uses the original dataset with temperature, humidity, and sound_volume as features.\
Step-by-step instructions to run the system and check accuracy.\
Run with Dataset v2 (Retraining & Threshold) –\
Demonstrates retraining the model with different features (e.g., light exposure, vibration).\
Shows how to adjust threshold to reach 1.0 anomaly recall after retraining.\

🛠 Tech Stack
Python\
Flask\
scikit-learn\
MLflow\
pandas, numpy\
joblib for model persistence\
