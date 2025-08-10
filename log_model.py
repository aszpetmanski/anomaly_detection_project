import mlflow.pyfunc
from wrapper import IsolationForestWrapper

# Include both the model and features.json in artifacts
artifacts = {
    "model_path": "model.pkl",
    "features_path": "features.json"
}

mlflow.pyfunc.save_model(
    path="mlflow_isolation_model",
    python_model=IsolationForestWrapper(),
    artifacts=artifacts,
    conda_env=None  # Skip conda for simplicity
)

print("Model wrapped with dynamic feature support and saved to 'mlflow_isolation_model'")
