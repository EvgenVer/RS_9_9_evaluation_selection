Homework for RS School Machine Learning course.

This project uses [Forest Cover Type](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset.

## Usage
This package allows you to train model for predicting the type of cover forest.
1. Clone this repository to your machine.
2. Download [Forest Cover Type](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine.
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. You can make EDA report:
```sh
poetry run eda_report -d <path to csv with data> -s <path to save report>
```

6. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
7. You can make some experements:
```sh
poetry run eval_perform
```

8. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```

![MLFlow experiments](https://drive.google.com/file/d/1qp3f4d6UEJnbyQG9Efvzk2YGc6SBrKsV/view?usp=sharing)