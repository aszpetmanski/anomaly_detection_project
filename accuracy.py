import os, json
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report

# Load data & features
df = pd.read_csv("model_training_dataset.csv")
with open("features.json","r") as f:
    FEATURES = json.load(f)

# Prepare X / y
X = df[FEATURES].copy()
has_labels = "anomaly" in df.columns
y_true = df["anomaly"].copy() if has_labels else None

# Load model
model = joblib.load("model.pkl")

# Optional scaler support (if you later save one)
scaler_path = "scaler.pkl"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    X = scaler.transform(X)

# Threshold (match wrapper): default 0.05; override via env
THRESH = float(os.getenv("ANOMALY_THRESHOLD", "0.05"))

# Scores: higher = more normal, lower = more anomalous
scores = model.decision_function(X)
y_pred = (scores < THRESH).astype(int)  # 1 = anomaly, 0 = normal

if has_labels:
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal","Anomaly"]))
else:
    print(f"No 'anomaly' labels. Predicted outlier rate: {(y_pred==1).mean():.4f}")






"""import pandas as pd
import joblib
import json
from sklearn.metrics import confusion_matrix, classification_report

# Load training/eval dataset and model
df = pd.read_csv("model_training_dataset.csv")
model = joblib.load("model.pkl")

# Load the exact feature list used during training
with open("features.json", "r") as f:
    FEATURES = json.load(f)

# Build X dynamically; keep only rows where all required features are present
X = df[FEATURES].copy()

# If labels exist, evaluate; otherwise just report outlier rate
has_labels = "anomaly" in df.columns

if has_labels:
    # Align y with any rows we might drop later
    y_true = df["anomaly"].copy()

    # (Optional) drop rows with missing features to avoid prediction errors
    mask = X.notna().all(axis=1)
    X = X[mask]
    y_true = y_true[mask]

    # Predict: IsolationForest returns -1 (anomaly) / 1 (normal)
    y_pred_raw = model.predict(X)
    y_pred = [1 if p == -1 else 0 for p in y_pred_raw]

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))
else:
    # No labels available: just show basic stats
    mask = X.notna().all(axis=1)
    X = X[mask]
    y_pred_raw = model.predict(X)
    outlier_rate = (y_pred_raw == -1).mean()
    print(f"No 'anomaly' column found. Predicted outlier rate on {len(X)} rows: {outlier_rate:.4f}")"""








"""import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset and model
df = pd.read_csv("model_training_dataset.csv")
model = joblib.load("model.pkl")

# Features only
X = df[['temperature', 'humidity', 'sound_volume']]

# True labels (from simulation): 1 = injected anomaly, 0 = normal
y_true = df['anomaly']

# Predict with Isolation Forest
# model.predict() returns:
#  -1 for anomaly
#   1 for normal
y_pred_raw = model.predict(X)

# Convert to same scale as y_true
y_pred = [1 if p == -1 else 0 for p in y_pred_raw]  # 1 = anomaly, 0 = normal

# Evaluate
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))"""

