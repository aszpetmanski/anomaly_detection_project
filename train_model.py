import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import json

# Load training dataset
df = pd.read_csv("model_training_dataset.csv")

# Drop the 'anomaly' column if present (unsupervised training)
if 'anomaly' in df.columns:
    df = df.drop(columns=['anomaly'])

# Select all feature columns dynamically (everything left)
features = list(df.columns)
X = df[features]

# Create and train Isolation Forest
model = IsolationForest(
    n_estimators=100,
    contamination='auto',  # or fixed fraction e.g., 0.05
    random_state=42
)
model.fit(X)

# Save model
joblib.dump(model, "model.pkl")

# Save features list
with open("features.json", "w") as f:
    json.dump(features, f)

print(f"Model trained and saved to model.pkl")
print(f"Feature list saved to features.json: {features}")
