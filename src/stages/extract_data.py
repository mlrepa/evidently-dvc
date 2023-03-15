import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Text
import yaml


def extract_data(config_path: Text) -> None:
    """Extract reference and current data by its dates.

    Args:
        config_path (Text): path to config
    """

    with open(config_path) as config_f:
        config: Dict = yaml.safe_load(config_f)

    logging.basicConfig(
        level=config['base']['logging_level'],
        format='EXTRACT_DATA: %(message)s'
    )

    workdir: Path = Path(config['base']['workdir'])

    logging.info('Load raw data')
    raw_data_path: Dict = workdir / config['data']['raw_data']
    raw_data: pd.DataFrame = pd.read_csv(
        raw_data_path,
        header=0,
        sep=',',
        parse_dates=['dteday'],
        index_col='dteday'
    )

    logging.info('Extract reference and current data')
    ref_dates_range: Text = config['extract_data']['ref_dates_range']
    cur_dates_range: Text = config['extract_data']['cur_dates_range']
    ref_start_date, ref_end_date = ref_dates_range.split('--')
    cur_start_date, cur_end_date = cur_dates_range.split('--')
    reference: pd.DataFrame = raw_data.loc[ref_start_date:ref_end_date]
    current: pd.DataFrame = raw_data.loc[cur_start_date:cur_end_date]

    logging.info('Save reference and current data')
    reference.to_csv(workdir / config['data']['reference_data'])
    current.to_csv(workdir / config['data']['current_data'])


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        '--config',
        dest='config',
        required=True
    )
    args = args_parser.parse_args()

    extract_data(
        config_path=Path(args.config)
    )
