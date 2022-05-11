#from pathlib import Path
import click
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

from .pipeline import create_pipeline

df = pd.read_csv("data/train.csv")
click.echo(f'Dataset shape: {df.shape}.')
features = df.drop('Cover_Type', axis=1)
target = df['Cover_Type']

models_type = ['logreg', 'forest']
logreg_params = [{
        'penalty': 'l2',
        'C': 1.0,
        'max_iter': 1500,
        'solver': 'lbfgs'
    },
    {
        'penalty': 'l1',
        'C': 10.0,
        'max_iter': 1500,
        'solver': 'liblinear'
    },
    {
        'penalty': 'l2',
        'C': 10.0,
        'max_iter': 1500,
        'solver': 'lbfgs'
    }]
forest_params = [{
        'n_estimators': 100,
        'criterion': 'gini',
        'max_depth': 5,
    },
    {
        'n_estimators': 200,
        'criterion': 'entropy',
        'max_depth': 10,
    },
    {
        'n_estimators': 300,
        'criterion': 'entropy',
        'max_depth': None,
    }]
engins = [(True, False), (False, True)]
scoring = ['accuracy', 'precision_macro', 'recall_macro']
random_state=42


def eval_perform() -> None:
    for model_type in models_type:
        for engin in engins:
            use_scaler, quant_transform = engin
            list_param = logreg_params if model_type == 'logreg' else forest_params
            for params in list_param:
                with mlflow.start_run():
                    pipeline = create_pipeline(use_scaler, model_type, random_state, params, quant_transform)
                    scores = cross_validate(pipeline, features, target, scoring=scoring, cv=5)
                    accuracy = np.mean(scores['test_accuracy'])
                    precision = np.mean(scores['test_precision_macro'])
                    recall = np.mean(scores['test_recall_macro'])
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