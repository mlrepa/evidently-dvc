import argparse
import logging
import pandas as pd
from pathlib import Path
import pickle
from typing import Dict, List, Text
import yaml


def predict(config_path: Text) -> None:
    """Make and save predictions on reference and current data.
    Args:
        config_path (Text): path to config
    """

    with open(config_path) as config_f:
        config: Dict = yaml.safe_load(config_f)
        
    logging.basicConfig(
        level=config['base']['logging_level'],
        format='PREDICT: %(message)s'
    )

    workdir: Path = Path(config['base']['workdir'])

    numerical_features: List[Text] = config['data']['numerical_features']
    categorical_features: List[Text] = config['data']['categorical_features']
    prediction_col: Text = config['data']['prediction_col']

    logging.info('Load reference and current data')
    reference_data_path: Path = workdir / config['data']['reference_data']
    current_data_path: Path = workdir / config['data']['current_data']
    reference: pd.DataFrame = pd.read_csv(
        reference_data_path,
        index_col='dteday'
    )
    current: pd.DataFrame = pd.read_csv(
        current_data_path,
        index_col='dteday'
    )
    
    logging.info('Load model')
    model_path: Path = workdir / config['predict']['model_path']
    with open(model_path, 'rb') as model_f:
        regressor = pickle.load(model_f)

    logging.info('Make predictions')
    ref_prediction = regressor.predict(reference[numerical_features + categorical_features])
    current_prediction = regressor.predict(current[numerical_features + categorical_features])

    logging.info('Save data with predictions')
    reference[prediction_col] = ref_prediction
    current[prediction_col] = current_prediction
    reference_prediction_path: Path = (
        workdir / config['data']['reference_prediction']
    )
    current_prediction_path: Path = (
        workdir / config['data']['current_prediction']
    )
    reference.to_csv(reference_prediction_path)
    current.to_csv(current_prediction_path)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        '--config',
        dest='config',
        required=True
    )
    args = args_parser.parse_args()

    predict(config_path=args.config)