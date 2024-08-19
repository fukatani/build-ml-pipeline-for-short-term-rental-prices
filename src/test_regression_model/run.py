#!/usr/bin/env python
"""
regression test for trained model against test dataset
"""
import argparse
import logging
import pandas
import wandb
import mlflow
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="test_regression_model")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Downloading and reading test artifact")
    test_data_path = run.use_artifact(args.test_dataset).file()
    df = pandas.read_csv(test_data_path, low_memory=False)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    X_test = df.copy()
    y_test = X_test.pop("price")

    logger.info("Downloading and reading the mlflow model")
    model_export_path = run.use_artifact(args.mlflow_model).download()
    pipe = mlflow.sklearn.load_model(model_export_path)

    y_pred = pipe.predict(X_test)

    logger.info("mae")
    mae = mean_absolute_error(y_test, y_pred)
    run.summary["mae"] = mae

    logger.info("r2")
    score = r2_score(y_test, y_pred)
    run.summary["mae"] = score

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="regression test for trained model")


    parser.add_argument(
        "--mlflow_model", 
        type=str,
        help="Test target model",
        required=True
    )

    parser.add_argument(
        "--test_dataset", 
        type=str,
        help="Dataset for test",
        required=True
    )


    args = parser.parse_args()

    go(args)
