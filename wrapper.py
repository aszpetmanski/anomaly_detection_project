import pandas as pd
import joblib
import mlflow.pyfunc
import json
import os

class IsolationForestWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load the model
        self.model = joblib.load(context.artifacts["model_path"])

        # Load the features list
        features_path = context.artifacts.get("features_path")
        if features_path and os.path.exists(features_path):
            with open(features_path, "r") as f:
                self.features = json.load(f)
        else:
            raise FileNotFoundError("features.json not found in artifacts.")

    def predict(self, context, model_input):
        # Ensure DataFrame
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        # Keep only expected features (ignore extras, fill missing with NaN)
        model_input = model_input.reindex(columns=self.features)

        # Predict
        scores = self.model.decision_function(model_input)
        raw_preds = self.model.predict(model_input)

        return pd.DataFrame({
            "anomaly_score": scores,
            "anomaly": [1 if p == -1 else 0 for p in raw_preds]
        })

