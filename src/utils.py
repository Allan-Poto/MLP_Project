from pathlib import Path

import sqlite3
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score
from config.core import DATA_DIR, MODEL_DIR, config


def load_from_db(database: str, query: str):
	"""
	Connect to database and load dataset

	Inputs:
		1. Database File Name
		2. Query to be used on the database

	Output:
		1. Pandas DataFrame of the Query
	"""
	conn = sqlite3.connect(Path(f'{DATA_DIR}/{database}'))
	df = pd.read_sql_query( f'{query}', conn)
	return df

def split_data(data: pd.DataFrame):
	"""
	Drop duplicated rows and returns the splitted dataset in X_train, X_test, y_train, y_test

	Inputs:
		1. Pandas DataFrame containing the data
	
	Outputs:
		1. X_train
		2. X_test
		3. y_train
		4. y_test
	"""
	data = data.copy()
	data = data.drop_duplicates(keep="first")
	X_train, X_test, y_train, y_test = train_test_split(
		data[data.columns[data.columns != config.TARGET_VARIABLE]],
		data[config.TARGET_VARIABLE],
		test_size=config.TEST_SIZE,
		random_state=config.SEED,
	)
	return X_train, X_test, y_train, y_test

def evaluate_model(data, metric):
	"""
	Evaluate the model training and testing score on the stated metrics.
	Available Metrics:
		1. f1score
		2. recall

	Inputs:
		1. data = [x_pred, y_train, y_pred, y_test]
		2. metric as above
	
	Outputs:
		1. Training_metric_score
		2. Testing_metric_score
	"""
	x_pred, y_train, y_pred, y_test = data
	if metric == "f1score":
		return f1_score(y_train, x_pred)*100, f1_score(y_test, y_pred)*100
	elif metric == "recall":
		return recall_score(y_train, x_pred)*100, recall_score(y_test, y_pred)*100
	else:
		raise ValueError("Add the metric to evaluate function in utils")


def save_pipeline(model_file: Pipeline):
	"""
	Save a pipeline in the model directory

	Input:
		1. Model Pipeline
	"""
	save_file_name = f"{config.MODEL_SELECTION}_{config.MODEL_VERSION}.pkl"
	save_path = f'{MODEL_DIR}/{save_file_name}'

	joblib.dump(model_file, save_path)


def load_pipeline(model_file: str) -> Pipeline:
	"""
	Load a pipeline from the model directory

	Input: 
		1. Model Pipeline File Name

	Output:
		1. Model Pipeline
	"""
	file_path = f'{MODEL_DIR}/{model_file}'
	trained_model = joblib.load(filename=file_path)
	return trained_model