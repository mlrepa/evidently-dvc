import argparse
import logging
from pathlib import Path
from typing import Dict, List, Text

import pandas as pd
import yaml
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.report import Report


def monitoring(config_path: Text) -> None:
    """Build and save data validation reports.

    Args:
        week_start (Text): Start date of a week.
        week_end (Text): End data of a week.
    """

    with open(config_path) as config_f:
        config: Dict = yaml.safe_load(config_f)
        print(config)

    logging.basicConfig(
        level=config["base"]["logging_level"], format="MONITORING: %(message)s"
    )

    WEEK_START = config["predict"]["week_start"]
    WEEK_END = config["predict"]["week_end"]
    logging.info(f"Predict for period: {WEEK_START} - {WEEK_END}")

    PREDICTIONS_DIR: Path = Path(config["predict"]["predictions_dir"])
    REPORTS_DIR: Path = Path(config["monitoring"]["reports_dir"]) / f"{WEEK_START}--{WEEK_END}"
    REPORTS_DIR.mkdir(exist_ok=True)

    # logging.info("Load data")
    reference_data_path: Path = Path(config["monitoring"]["reference_data"])
    reference: pd.DataFrame = pd.read_csv(reference_data_path, index_col="dteday").reset_index()
    current_data_path: Path = PREDICTIONS_DIR / f"{WEEK_START}--{WEEK_END}.csv"
    current: pd.DataFrame = pd.read_csv(current_data_path, index_col="dteday")
    current = current.loc[WEEK_START:WEEK_END].reset_index()

    logging.info("Prepare column_mapping object for Evidently reports")
    column_mapping = ColumnMapping()
    column_mapping.numerical_features: List[Text]  = config["data"]["numerical_features"]

    logging.info("Build Data quality report...")
    
    # Data Quality
    data_quality_report = Report(metrics=[DataQualityPreset()])
    data_quality_report.run(
        reference_data=reference, 
        current_data=current, 
        column_mapping=column_mapping
    )
    data_quality_path = REPORTS_DIR / config["monitoring"]["data_quality_path"]
    data_quality_report.save_html(str(data_quality_path))
    logging.info(f"Data quality report saved to: {data_quality_path}")

    # Data Drift
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(
        reference_data=reference, current_data=current, column_mapping=column_mapping
    )
    data_drift_path = REPORTS_DIR / config["monitoring"]["data_drift_path"]
    data_drift_report.save_html(str(data_drift_path))
    logging.info(f"Data drift report saved to: {data_drift_path}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    monitoring(config_path=args.config)
