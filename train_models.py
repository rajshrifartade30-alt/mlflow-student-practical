import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save plots without opening a GUI window.
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    base_dir = Path(__file__).resolve().parent
    tracking_dir = base_dir / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    # Create a Windows-safe local MLflow tracking URI.
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment("Student_Performance_Regression")

    # Generate a reproducible student-style regression dataset.
    X, y = make_regression(
        n_samples=300,
        n_features=6,
        n_informative=5,
        noise=15.0,
        random_state=42,
    )

    columns = [f"feature_{i}" for i in range(1, 7)]
    df = pd.DataFrame(X, columns=columns)
    df["target_marks"] = y

    data_path = base_dir / "student_dataset.csv"
    df.to_csv(data_path, index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        df[columns],
        df["target_marks"],
        test_size=0.2,
        random_state=42,
    )

    models = {
        "LinearRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            random_state=42,
        ),
    }

    results = []

    for name, model in models.items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            mse = float(mean_squared_error(y_test, predictions))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_test, predictions))
            r2 = float(r2_score(y_test, predictions))

            mlflow.log_param("model_type", name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            if name == "RandomForest":
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("max_depth", 8)

            mlflow.log_metrics(
                {
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                }
            )

            mlflow.log_artifact(
                str(data_path),
                artifact_path="dataset",
            )

            # MLflow uploads these files before the temporary directory closes.
            with tempfile.TemporaryDirectory() as temp_dir:
                metrics_path = Path(temp_dir) / "metrics.json"
                metrics_path.write_text(
                    json.dumps(
                        {
                            "mse": mse,
                            "rmse": rmse,
                            "mae": mae,
                            "r2": r2,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                mlflow.log_artifact(
                    str(metrics_path),
                    artifact_path="evaluation",
                )

                plot_path = Path(temp_dir) / "actual_vs_predicted.png"
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(y_test, predictions)
                ax.set_xlabel("Actual target")
                ax.set_ylabel("Predicted target")
                ax.set_title(f"Actual vs Predicted: {name}")
                fig.tight_layout()
                fig.savefig(plot_path, dpi=150)
                plt.close(fig)

                mlflow.log_artifact(
                    str(plot_path),
                    artifact_path="plots",
                )

            signature = infer_signature(
                X_train,
                model.predict(X_train),
            )
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                input_example=X_train.head(3),
                signature=signature,
            )

            results.append(
                {
                    "run_id": run.info.run_id,
                    "model_name": name,
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                }
            )

            print(
                f"{name}: RMSE={rmse:.3f}, "
                f"MAE={mae:.3f}, R2={r2:.3f}"
            )

    results_df = pd.DataFrame(results).sort_values("rmse")
    results_path = base_dir / "experiment_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\nExperiment comparison (lower RMSE is better):")
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    best_path = base_dir / "best_model_run.txt"
    best_path.write_text(
        (
            f"Best model by RMSE: {best['model_name']}\n"
            f"Run ID: {best['run_id']}\n"
            f"RMSE: {best['rmse']:.6f}\n"
        ),
        encoding="utf-8",
    )

    print(f"\nBest model: {best['model_name']}")
    print(f"Results saved to {results_path.name}")
    print(f"Best run summary saved to {best_path.name}")
    print(f"MLflow tracking data saved in {tracking_dir}")


if __name__ == "__main__":
    main()