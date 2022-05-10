import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from pathlib import Path
import click

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
    default="data/report.html",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True, 
)

def eda_report(
    dataset_path: Path,
    save_model_path: Path
) -> None:
    df = pd.read_csv(dataset_path)
    profile = ProfileReport(df, title="Profiling report", explorative=True)
    profile.to_file(save_model_path)
    


