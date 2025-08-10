# IoT Anomaly Detection – Quick Start

## 1) Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## 2) Generate dataset
```bash
python dataset.py
```

## 3) Train model
```bash
python train_model.py
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


## 6) Serve model
```bash
mlflow models serve -m mlflow_isolation_model --no-conda -h 127.0.0.1 -p 5001

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