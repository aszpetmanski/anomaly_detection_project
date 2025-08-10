# IoT Anomaly Detection – Retrain with New Data & Threshold Tuning

## 0) Clean Previous Runs
Before starting, remove all old datasets, models, and MLflow runs to avoid conflicts:
```bash
rm -rf factory_sensor_data_with_anomalies.csv
rm -rf streaming_dataset.csv
rm -rf model_training_dataset.csv
rm -rf model.pkl
rm -rf mlflow_isolation_model
rm -rf anomaly_log.csv
rm -rf mlruns
```

## 1) Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## 2) Generate dataset
```bash
python dataset_v2.py
```

## 3) Train model
```bash
python train_model.py
```

## 3a) Accuracy check before tuning
```bash
python accuracy.py
```

## 4) Log model for MLflow
```bash
python log_model.py
```
## 5) Start data stream
```bash
python stream.py
```
Visit (in that order):\
http://127.0.0.1:5000/start \
http://127.0.0.1:5000/current


## 6) Serve model with custom threshold
```bash
ANOMALY_THRESHOLD=0.15 mlflow models serve -m mlflow_isolation_model --no-conda -h 127.0.0.1 -p 5001

```

## 7) Run connector
```bash
python stream_to_model.py

```
## 8) Log anomalies
```bash
python log_anomalies.py
```
## 9) Accuracy check
```bash
python accuracy.py
```