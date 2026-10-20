import os

import mlflow
from mlflow import MlflowClient


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tracking_dir = os.path.join(base_dir, "mlruns")

    # Build a local file URI that works on Windows.
    tracking_uri = "file:///" + tracking_dir.replace("\\", "/")
    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(
        "Student_Performance_Regression"
    )

    if experiment is None:
        raise SystemExit("Run python train_models.py first.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
    )

    if runs.empty:
        raise SystemExit("No experiment runs found.")

    best = runs.iloc[0]
    run_id = best["run_id"]
    model_uri = f"runs:/{run_id}/model"
    model_name = "StudentPerformanceBestModel"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    print("Registered model:", registered.name)
    print("Version:", registered.version)
    print("Selected run:", run_id)
    print("RMSE:", best["metrics.rmse"])

    client = MlflowClient()

    try:
        client.set_registered_model_alias(
            model_name,
            "champion",
            str(registered.version),
        )
        print("Alias champion set to version", registered.version)
    except Exception as exc:
        print(
            "Model registered; alias not available in this "
            f"MLflow version: {exc}"
        )


if __name__ == "__main__":
    main()