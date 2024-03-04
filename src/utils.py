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
    save_file_name = f"{config.MODEL_SELECTION}_{config.MODEL_VERSION}.pkl"
    save_path = f'{MODEL_DIR}/{save_file_name}'

    joblib.dump(model_file, save_path)


def load_pipeline(model_file: str) -> Pipeline:
    """
    Load a pipeline from the model directory
    """
    file_path = f'{MODEL_DIR}/{model_file}'
    trained_model = joblib.load(filename=file_path)
    return trained_model