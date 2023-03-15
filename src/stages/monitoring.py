import argparse
from evidently.dashboard import Dashboard
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.dashboard.tabs import (
    DataDriftTab,
    NumTargetDriftTab
)
from evidently.metrics import (
    RegressionQualityMetric,
    RegressionPredictedVsActualScatter,
    RegressionPredictedVsActualPlot,
    RegressionErrorPlot,
    RegressionAbsPercentageErrorPlot,
    RegressionErrorDistribution,
    RegressionErrorNormality,
    RegressionTopErrorMetric,
    RegressionErrorBiasTable,
    
    DatasetSummaryMetric,
    ColumnSummaryMetric,
    DatasetMissingValuesMetric,
    DatasetCorrelationsMetric
)
from evidently.report import Report
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Text
import yaml


def monitoring(config_path: Text) -> None:
    """Build and save monitoring reports.

    Args:
        week_start (Text): Start date of a week.
        week_end (Text): End data of a week.
    """

    with open(config_path) as config_f:
        config: Dict = yaml.safe_load(config_f)
        
    logging.basicConfig(
        level=config['base']['logging_level'],
        format='MONITORING: %(message)s'
    )

    workdir: Path = Path(config['base']['workdir'])

    week_start: Text = config['monitoring']['week_start']
    week_end: Text = config['monitoring']['week_end']
    reports_dir: Path = (
        workdir / 
        config['monitoring']['reports_dir'] / 
        f'{week_start}_{week_end}'
    )
    reports_dir.mkdir(exist_ok=True)
    
    logging.info('Load metadata')

    numerical_features: List[Text] = config['data']['numerical_features']
    categorical_features: List[Text] = config['data']['categorical_features']
    target_col: Text = config['data']['target_col']
    prediction_col: Text = config['data']['prediction_col']

    logging.info('Load reference and current data')
    reference_data_path: Path = workdir / config['data']['reference_prediction']
    current_data_path: Path = workdir / config['data']['current_prediction']
    reference: pd.DataFrame = pd.read_csv(
        reference_data_path,
        index_col='dteday'
    )
    current: pd.DataFrame = pd.read_csv(
        current_data_path,
        index_col='dteday'
    )

    column_mapping = ColumnMapping()
    column_mapping.target = target_col
    column_mapping.prediction = prediction_col
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features
    print(f'column mapping: {column_mapping}')

    logging.info('Build and save reports')

    logging.info('Model performance report...')
    model_performance_config: Dict = config['monitoring']['model_performance']
    regression_performance_report_dir: Path = (
        reports_dir / 
        model_performance_config['dir_name']
    )
    regression_performance_report_dir.mkdir(exist_ok=True)
    
    # Quality metric
    regression_quality_metric_report: Report = Report(metrics=[
        RegressionQualityMetric()
    ])
    regression_quality_metric_report.run(
        reference_data=reference,
        current_data=current.loc[week_start:week_end],
        column_mapping=column_mapping
    )
    regression_quality_metric_report_path: Path = (
        regression_performance_report_dir / 
        model_performance_config['quality_metric']
    )
    regression_quality_metric_report.save_html(
        regression_quality_metric_report_path
    )

    # Predict vs actual
    regression_predicted_vs_actual_report: Report = Report(metrics=[
        RegressionPredictedVsActualScatter(),
        RegressionPredictedVsActualPlot()
    ])
    regression_predicted_vs_actual_report.run(
        reference_data=reference,
        current_data=current.loc[week_start:week_end],
        column_mapping=column_mapping
    )
    regression_predicted_vs_actual_report_path: Path = (
        regression_performance_report_dir / 
        model_performance_config['predicted_vs_actual']
    )
    regression_predicted_vs_actual_report.save_html(
        regression_predicted_vs_actual_report_path
    )

    # Errors
    regression_errors_report: Report = Report(metrics=[
        RegressionErrorPlot(),
        RegressionAbsPercentageErrorPlot(),
        RegressionErrorDistribution(),
        RegressionErrorNormality(),
        RegressionTopErrorMetric(),
        RegressionErrorBiasTable()
    ])
    regression_errors_report.run(
        reference_data=reference,
        current_data=current.loc[week_start:week_end],
        column_mapping=column_mapping
    )
    regression_errors_report_path: Path = (
        regression_performance_report_dir / 
        model_performance_config['errors']
    )
    regression_errors_report.save_html(regression_errors_report_path)

    logging.info(
        f'Model performance report saved to: {regression_performance_report_dir}'
    )

    logging.info('Target drift report...')
    target_drift_dashboard = Dashboard(tabs=[NumTargetDriftTab()])
    target_drift_dashboard.calculate(
        reference,
        current.loc[week_start:week_end], 
        column_mapping=column_mapping
    )
    target_drift_report_path: Path = (
        reports_dir / config['monitoring']['target_drift']
    )
    target_drift_dashboard.save(target_drift_report_path)
    logging.info(
        f'Target drift report saved to: {target_drift_report_path}'
    )

    logging.info('Data drift report...')
    data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
    data_drift_dashboard.calculate(
        reference,
        current.loc[week_start:week_end],
        column_mapping=column_mapping
    )
    data_drift_report_path = reports_dir / config['monitoring']['data_drift']
    data_drift_dashboard.save(data_drift_report_path)
    logging.info(
        f'Data drift report saved to: {data_drift_report_path}'
    )

    logging.info('Data quality report...')
    data_quality_config: Dict = config['monitoring']['data_quality']
    data_quality_report_dir = reports_dir / data_quality_config['dir_name']
    data_quality_report_dir.mkdir(exist_ok=True)

    # Data summary
    data_summary_report: Report = Report(metrics=[
        DatasetSummaryMetric()
    ])
    data_summary_report.run(
        reference_data=reference,
        current_data=current.loc[week_start:week_end],
        column_mapping=column_mapping
    )
    data_summary_report_path: Path = (
        data_quality_report_dir / data_quality_config['data_summary']
    )
    data_summary_report.save_html(data_summary_report_path)

    # Column summary
    column_summary_report: Report = Report(metrics=[
        ColumnSummaryMetric(column_name=col)
        for col in numerical_features
    ])
    column_summary_report.run(
        reference_data=reference,
        current_data=current.loc[week_start:week_end],
        column_mapping=column_mapping
    )
    column_summary_report_path: Path = (
        data_quality_report_dir / data_quality_config['column_summary']
    )
    column_summary_report.save_html(column_summary_report_path)

    # Data correlation
    data_correlation_report: Report = Report(metrics=[
        DatasetCorrelationsMetric()
    ])
    data_correlation_report.run(
        reference_data=reference,
        current_data=current.loc[week_start:week_end],
        column_mapping=column_mapping
    )
    data_correlation_report_path: Path = (
        data_quality_report_dir / data_quality_config['data_correlation']
    )
    data_correlation_report.save_html(data_correlation_report_path)

    # Missing values
    missing_values_report: Report = Report(metrics=[
        DatasetMissingValuesMetric()
    ])
    missing_values_report.run(
        reference_data=reference,
        current_data=current.loc[week_start:week_end],
        column_mapping=column_mapping
    )
    missing_values_report_path: Path = (
        data_quality_report_dir / data_quality_config['missing_values']
    )
    missing_values_report.save_html(missing_values_report_path)

    logging.info(
        f'Data quality report saved to: {data_quality_report_dir}'
    )

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        '--config',
        dest='config',
        required=True
    )
    args = args_parser.parse_args()

    monitoring(config_path=args.config)
