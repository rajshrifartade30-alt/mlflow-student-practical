import os

import mlflow
import pandas as pd
from mlflow import MlflowClient


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tracking_dir = os.path.join(base_dir, "mlruns")

    # Build a local file URI that works on Windows.
    tracking_uri = "file:///" + tracking_dir.replace("\\", "/")
    mlflow.set_tracking_uri(tracking_uri)

    model_name = "StudentPerformanceBestModel"
    client = MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise SystemExit("Register the model first.")

    latest = max(versions, key=lambda version: int(version.version))
    model_uri = f"models:/{model_name}/{latest.version}"
    model = mlflow.pyfunc.load_model(model_uri)

    sample = pd.DataFrame(
        [
            {
                "feature_1": 0.2,
                "feature_2": -0.1,
                "feature_3": 0.5,
                "feature_4": 1.0,
                "feature_5": -0.4,
                "feature_6": 0.3,
            }
        ]
    )

    prediction = model.predict(sample)

    print("Model version:", latest.version)
    print("Sample prediction:", prediction[0])


if __name__ == "__main__":
    main()