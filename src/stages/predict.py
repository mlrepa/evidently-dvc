import argparse
import joblib
import logging
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, List, Text
import yaml


def predict(config_path: Text) -> None:
    """Make and save predictions on reference and predict data.
    Args:
        config_path (Text): path to config
    """

    with open(config_path) as config_f:
        config: Dict = yaml.safe_load(config_f)

    logging.basicConfig(
        level=config["base"]["logging_level"], format="PREDICT: %(message)s"
    )

    WEEK_START = config["predict"]["week_start"]
    WEEK_END = config["predict"]["week_end"]
    logging.info(f"Predict for period: {WEEK_START} - {WEEK_END}")

    workdir: Path = Path(config["base"]["workdir"])
    predictions_dir: Path = workdir / config["predict"]["predictions_dir"]
    predictions_dir.mkdir(exist_ok=True)

    logging.info("Load metadata")
    numerical_features: List[Text] = config["data"]["numerical_features"]
    categorical_features: List[Text] = config["data"]["categorical_features"]
    prediction_col: Text = config["data"]["prediction_col"]

    logging.info("Load data")
    predict_data_path: Path = workdir / config["data"]["predict_data"]
    predict_data: pd.DataFrame = pd.read_csv(predict_data_path, index_col="dteday")
    predict_data = predict_data[WEEK_START:WEEK_END]
    print(predict_data.shape)

    # reference_data_path: Path = workdir / config['data']['reference_data']
    # reference: pd.DataFrame = pd.read_csv(
    #     reference_data_path,
    #     index_col='dteday'
    # )

    logging.info("Load model")
    model_path: Path = workdir / config["predict"]["model_path"]
    model = joblib.load(model_path)
    print(model)

    logging.info("Make predictions")

    # ref_prediction = model.predict(reference[numerical_features + categorical_features])
    predictions = model.predict(predict_data[numerical_features + categorical_features])

    logging.info("Save data with predictions")
    # reference[prediction_col] = ref_prediction
    predict_data[prediction_col] = predictions
    # reference_prediction_path: Path = (
    #     workdir / config['data']['reference_prediction']
    # )
    predictions_path: Path = predictions_dir / f"{WEEK_START}--{WEEK_END}.csv"
    print(predictions_path)
    # reference.to_csv(reference_prediction_path)
    predict_data.to_csv(predictions_path)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    # args_parser.add_argument('--model', dest='model',required=True)
    # args_parser.add_argument('--data', dest='data',required=True)

    # args_parser.add_argument('--week_start', dest='week_start',required=True)
    # args_parser.add_argument('--week_end', dest='week_end',required=True)
    args = args_parser.parse_args()

    predict(config_path=args.config)
