from pathlib import Path
from typing import List

import sqlite3
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from config.core import DATA_DIR, MODEL_DIR, config


def load_from_db(database: str, query: str):
    """
    Connect to database and load dataset
    """
    conn = sqlite3.connect(Path(f'{DATA_DIR}/{database}'))
    df = pd.read_sql_query( f'{query}', conn)
    return df


def save_pipeline(model_file: Pipeline):
    """
    Save a pipeline in the model directory
    """
    save_file_name = f"{config.MODEL_VERSION}.pkl"
    save_path = f'{MODEL_DIR}/{save_file_name}'

    remove_pipelines(files_to_keep=[save_file_name])
    joblib.dump(model_file, save_path)


def load_pipeline(model_file: str) -> Pipeline:
    """
    Load a pipeline from the model directory
    """
    file_path = f'{MODEL_DIR}/{model_file}'
    print(file_path)
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_pipelines(files_to_keep: List[str]):
    """
    Remove all other models in the model directory
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()