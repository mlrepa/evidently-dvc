import argparse
import logging
import pandas as pd
from pathlib import Path
import pickle
from sklearn import ensemble
from typing import Dict, List, Text
import yaml


def train(config_path: Text) -> None:
    """Train model.

    Args:
        config_path (Text): path to config
    """

    with open(config_path) as config_f:
        config: Dict = yaml.safe_load(config_f)
        
    logging.basicConfig(
        level=config['base']['logging_level'],
        format='TRAIN: %(message)s'
    )

    workdir: Path = Path(config['base']['workdir'])

    numerical_features: List[Text] = config['data']['numerical_features']
    categorical_features: List[Text] = config['data']['categorical_features']
    target_col: Text = config['data']['target_col']

    logging.info('Load reference data')

    reference_data_path: Path = workdir / config['data']['reference_data']
    reference: pd.DataFrame = pd.read_csv(
        reference_data_path,
        index_col='dteday'
    )

    logging.info('Train model')
    regressor = ensemble.RandomForestRegressor(
        random_state=0, 
        n_estimators=config['train']['n_estimators']
    )
    regressor.fit(
        reference[numerical_features + categorical_features],
        reference[target_col]
    )

    logging.info('Dump model')
    model_path: Path = workdir / config['train']['model_path']
    with open(model_path, 'wb') as model_f:
        pickle.dump(regressor, model_f)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        '--config',
        dest='config',
        required=True
    )
    args = args_parser.parse_args()

    train(config_path=args.config)
