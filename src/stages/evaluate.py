import argparse
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Text

from dvclive import Live
import joblib
import pandas as pd
import yaml
from evidently import ColumnMapping
from evidently.metrics import (      
    RegressionQualityMetric,
    RegressionPredictedVsActualScatter,
    RegressionErrorNormality,
    RegressionTopErrorMetric,
    RegressionDummyMetric,
)
from evidently.report import Report

def numpy_to_standard_types(input_data: Dict) -> Dict:
    """Convert numpy type values to standard Python types in flat(!) dictionary.

    Args:
        input_data (Dict): Input data (flat dictionary).

    Returns:
        Dict: Dictionary with standard value types.
    """

    output_data: Dict = {}

    for k, v in input_data.items():
        if isinstance(v, np.generic):
            v = v.item()
        output_data[k] = v

    return output_data



def evaluate(config_path: Text) -> None:
    """Train model.

    Args:
        config_path (Text): path to config
    """

    with open(config_path) as config_f:
        config: Dict = yaml.safe_load(config_f)

    logging.basicConfig(
        level=config["base"]["logging_level"], format="TRAIN: %(message)s"
    )

    WORKDIR: Path = Path(config["base"]["workdir"])
    REPORTS_DIR: Path = WORKDIR / config["base"]["reports_dir"]
    REPORTS_DIR.mkdir(exist_ok=True)

    numerical_features: List[Text] = config["data"]["numerical_features"]
    categorical_features: List[Text] = config["data"]["categorical_features"]
    target_col: Text = config["data"]["target_col"]
    prediction_col: Text = config["data"]["prediction_col"]

    logging.info("Load data")
    train_data_path: Path = WORKDIR / config["data"]["train_data"]
    test_data_path: Path = WORKDIR / config["data"]["test_data"]
    train_data: pd.DataFrame = pd.read_csv(train_data_path, index_col="dteday")
    test_data: pd.DataFrame = pd.read_csv(test_data_path, index_col="dteday")

    logging.info("Load model")
    model_path: Path = WORKDIR / config["train"]["model_path"]
    model = joblib.load(model_path)

    logging.info("Get predictions to TEST data")
    train_prediction = model.predict(
        train_data[numerical_features + categorical_features]
    )
    test_prediction = model.predict(
        test_data[numerical_features + categorical_features]
    )

    logging.info("Prepare datasets for monitoring)")
    test_data["prediction"] = test_prediction
    train_data["prediction"] = train_prediction
    reference_data = train_data.sample(frac=0.3)

    logging.info("Prepare column_mapping object for Evidently reports")
    column_mapping = ColumnMapping()
    column_mapping.target = target_col
    column_mapping.prediction = prediction_col
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    logging.info("Create a model performance report")
    model_performance_report = Report(
        metrics=[
            RegressionQualityMetric(),
            RegressionPredictedVsActualScatter(),
            RegressionErrorNormality(),
            RegressionTopErrorMetric(),
            RegressionDummyMetric(),
        ]
    )

    logging.info("Calculate metrics")
    model_performance_report.run(
        reference_data=reference_data,
        current_data=test_data,
        column_mapping=column_mapping,
    )
    
    logging.info("Extract metrics")
    regression_metrics: Dict = model_performance_report.as_dict()['metrics'][0]['result']["current"]
    metric_names = ['r2_score', 'rmse', 'mean_error', 'mean_abs_error', 'mean_abs_perc_error']
    selected_metrics = {k: regression_metrics.get(k) for k in metric_names} 
    
    logging.info("Save evaluation metrics and model report")
    with Live(dir=REPORTS_DIR, 
              save_dvc_exp=False, 
              dvcyaml="dvc.yaml",) as live:

        # Log metrics 
        [live.log_metric(k, v, plot=False) for k,v in selected_metrics.items()]

        # Save reports in HTML format
        model_performance_report_path = REPORTS_DIR / "model_performance.html"
        model_performance_report.save_html(str(model_performance_report_path))
        # live.log_artifact(
        #     model_performance_report_path,
        #     name="model_performance_report",
        #     desc="Model performance report",
        #     labels=["train", "model_performance"],
        #     cache=False
        # )

    logging.info("Save reference data")
    ref_data_path = WORKDIR / config["evaluate"]["reference_data"]
    reference_data.to_csv(ref_data_path)
    logging.info(f"Saved reference data to to {ref_data_path}")


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    evaluate(config_path=args.config)
