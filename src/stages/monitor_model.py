import argparse
import logging
from pathlib import Path
from typing import Dict, List, Text

import pandas as pd
import yaml
from evidently import ColumnMapping
from evidently.metric_preset import RegressionPreset, TargetDriftPreset
from evidently.report import Report


def monitoring(config_path: Text) -> None:
    """Build and save monitoring reports.

    Args:
        week_start (Text): Start date of a week.
        week_end (Text): End data of a week.
    """

    with open(config_path) as config_f:
        config: Dict = yaml.safe_load(config_f)

    logging.basicConfig(
        level=config["base"]["logging_level"], format="MONITORING: %(message)s"
    )

    WEEK_START = config["predict"]["week_start"]
    WEEK_END = config["predict"]["week_end"]
    logging.info(f"Predict for period: {WEEK_START} - {WEEK_END}")

    workdir: Path = Path(config["base"]["workdir"])
    predictions_dir: Path = workdir / config["predict"]["predictions_dir"]
    reports_dir: Path = (
        workdir / config["monitoring"]["reports_dir"] / f"{WEEK_START}--{WEEK_END}"
    )
    reports_dir.mkdir(exist_ok=True)
    logging.info(f"Predict for period: {WEEK_START} - {WEEK_END}")

    logging.info("Load metadata")
    numerical_features: List[Text] = config["data"]["numerical_features"]
    categorical_features: List[Text] = config["data"]["categorical_features"]
    target_col: Text = config["data"]["target_col"]
    prediction_col: Text = config["data"]["prediction_col"]

    logging.info("Load data")
    reference_data_path: Path = workdir / config["monitoring"]["reference_data"]
    reference: pd.DataFrame = pd.read_csv(reference_data_path, index_col="dteday")

    current_data_path: Path = predictions_dir / f"{WEEK_START}--{WEEK_END}.csv"
    current: pd.DataFrame = pd.read_csv(current_data_path, index_col="dteday")
    current = current.loc[WEEK_START:WEEK_END]

    logging.info("Prepare column_mapping object for Evidently reports")
    column_mapping = ColumnMapping()
    column_mapping.target = target_col
    column_mapping.prediction = prediction_col
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features
    print(f"column mapping: {column_mapping}")

    logging.info("Create a model performance report...")
    model_performance_report = Report(metrics=[RegressionPreset()])
    model_performance_report.run(
        reference_data=reference, current_data=current, column_mapping=column_mapping
    )
    model_performance_report_path = (
        reports_dir / config["monitoring"]["model_performance_path"]
    )
    model_performance_report.save_html(model_performance_report_path)
    logging.info(f"Regression report saved to {model_performance_report_path}")

    logging.info("Target drift report...")
    target_drift_report = Report(metrics=[TargetDriftPreset()])
    target_drift_report.run(
        reference_data=reference, current_data=current, column_mapping=column_mapping
    )
    target_drift_report_path: Path = (
        reports_dir / config["monitoring"]["target_drift_path"]
    )
    target_drift_report.save_html(target_drift_report_path)
    logging.info(f"Target drift report saved to: {target_drift_report_path}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    monitoring(config_path=args.config)
