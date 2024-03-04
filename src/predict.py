import pandas as pd
from sklearn.pipeline import Pipeline

from config.core import config
from utils import load_from_db, split_data, load_pipeline, evaluate_model


def make_prediction(model: Pipeline, X_test: pd.DataFrame):
	"""
	Make a prediction using the saved model

	Inputs:
		1. Model Pipeline
		2. X_test
	
	Output:
		1. Dictionary with the following keys:
			a. "predictions" : Contains the result
			b. "model" : Type of Model used
			c. "version" : Version of Model used
	"""

	predictions = model.predict(X_test)
	results = {
		"predictions": predictions,
		"model": config.MODEL_SELECTION,
		"version": config.MODEL_VERSION,
	}
	return results

if __name__ == "__main__":
	"""
	Loads a model based on configurations and make predictions, output prediction score

	TODO: Store the result in a separate location
	"""
	# Load Data
	data = load_from_db(config.DATA, config.QUERY)
	X_train, X_test, y_train, y_test = split_data(data)

	# Load Model
	pipeline_file_name = f"{config.MODEL_SELECTION}_{config.MODEL_VERSION}.pkl"
	model = load_pipeline(model_file=pipeline_file_name)

	# Make Predictions
	x_pred = make_prediction(model, X_train)["predictions"]
	y_pred = make_prediction(model, X_test)["predictions"]

	# Evaluate Model
	Training_F1_score, Testing_F1_score = evaluate_model([x_pred, y_train, y_pred, y_test], "f1score")
	print(f"Training F1_Score: {Training_F1_score:.2f}%")
	print(f"Testing F1_Score: {Testing_F1_score:.2f}%")

	Training_recall, Testing_recall = evaluate_model([x_pred, y_train, y_pred, y_test], "recall")
	print(f"Training Recall: {Training_recall:.2f}%")
	print(f"Testing Recall: {Testing_recall:.2f}%")