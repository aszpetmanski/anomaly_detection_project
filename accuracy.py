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
