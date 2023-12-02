import argparse
import logging
from pathlib import Path
from typing import Dict, Text

import pandas as pd
import yaml


def extract_data(config_path: Text) -> None:
    """Extract reference and current data by its dates.

    Args:
        config_path (Text): path to config
    """

    with open(config_path) as config_f:
        config: Dict = yaml.safe_load(config_f)

    logging.basicConfig(
        level=config["base"]["logging_level"], format="EXTRACT_DATA: %(message)s"
    )

    logging.info("Load raw data")
    raw_data_path: Dict = Path(config["data"]["raw_data"])
    raw_data: pd.DataFrame = pd.read_csv(
        raw_data_path, header=0, sep=",", parse_dates=["dteday"], index_col="dteday"
    )

    logging.info("Extract reference and current data")
    train_dates_range: Text = config["extract_data"]["train_dates_range"]
    test_dates_range: Text = config["extract_data"]["test_dates_range"]

    TRAIN_START, TRAIN_END = train_dates_range.split("--")
    TEST_START, TEST_END = test_dates_range.split("--")
    train_data: pd.DataFrame = raw_data.loc[TRAIN_START:TRAIN_END]
    test_data: pd.DataFrame = raw_data.loc[TEST_START:TEST_END]
    
    print(type(train_data))

    logging.info("Save train_data and test_data data")
    train_data.to_csv(config["data"]["train_data"])
    test_data.to_csv(config["data"]["test_data"])


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest="config", required=True)
    args = args_parser.parse_args()

    extract_data(config_path=args.config)
