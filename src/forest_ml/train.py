from pathlib import Path
from joblib import dump
import click
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score

from .data import get_data
from .pipeline import create_pipeline

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True, 
)
@click.option(
    "-t",
    "--model-type",
    default="logreg",
    type=str,
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "-p",
    "--params",
    default={
        'penalty': 'l2',
        'C': 1.0,
        'max_iter': 1000,
    },
    type=dict,
    show_default=True,
)
@click.option(
    "--quant-transform",
    default=False,
    type=bool,
    show_default=True,
)

def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    model_type: str,
    params: dict,
    quant_transform: bool,
) -> None:
    features_train, features_val, target_train, target_val = get_data(
        dataset_path, random_state, test_split_ratio,
    )
    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler, model_type, random_state, params, quant_transform)
        pipeline.fit(features_train, target_train)
        predict = pipeline.predict(features_val)
        accuracy = accuracy_score(target_val, predict)
        precision = precision_score(target_val, predict, average='weighted')
        recall = recall_score(target_val, predict, average='weighted')
        mlflow.log_param('use_scaler', use_scaler)
        mlflow.log_param('quant_transfor', quant_transform)
        mlflow.log_param('model type', model_type)
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric('Accuracy', accuracy)
        mlflow.log_metric('Precision', precision)
        mlflow.log_metric('recall', recall)
        click.echo(f'Accuracy: {accuracy}')
        click.echo(f'Precision: {precision}')
        click.echo(f'Recall: {recall}')
        dump(pipeline, save_model_path)
        click.echo(f'Model is saved to {save_model_path}')


