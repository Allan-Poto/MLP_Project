import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from config.core import config
from pipeline import model_pipeline, tuning_pipeline
from utils import load_from_db, split_data, save_pipeline, evaluate_model


def train_model(data, optimize: bool=False):
	"""
	Trains a model using the pipeline.
	If optimize is set to True, the model runs the hyperparameter tuning pipeline instead and
	prints out the following:
		1. Best params
		2. Training recall and f1_score using Best params
		3. Testing recall and f1_score using Best params
	
	# TODO: Add functionality and Pipeline to train and Save a model using the optimized hyperparameters 
		at the end of the optimization
	"""
	X_train, X_test, y_train, y_test = data

	# fit model
	if optimize:
		if config.MODEL_SELECTION == "LogisticRegression":
			param_grid = config.LOG_TUNING_STR_GRID
			param_grid.update(config.LOG_TUNING_INT_GRID)
			param_grid.update(config.LOG_TUNING_FLOAT_GRID)
		elif config.MODEL_SELECTION == "LGBMClassifier":
			param_grid = config.LGBM_TUNING_INT_GRID
			param_grid.update(config.LGBM_TUNING_FLOAT_GRID)
			# param_grid.update(config.LGBM_TUNING_STR_GRID)
		elif config.MODEL_SELECTION == "SVC":
			param_grid = config.SVC_TUNING_FLOAT_GRID
			param_grid.update(config.SVC_STR_HPARAMS)
		elif config.MODEL_SELECTION == "MLPClassifier":
			param_grid = config.TUNING_MLP_STR_HPARAMS
			param_grid.update(config.TUNING_MLP_INT_HPARAMS)
			param_grid.update(config.TUNING_MLP_FLOAT_HPARAMS)
		else:
			raise ValueError("Model does not exist")
		rscv = RandomizedSearchCV(tuning_pipeline, param_grid, cv=config.CV, scoring="accuracy")
		rscv.fit(X_train, y_train)
		
		print(f'Best Hyperparameters: {rscv.best_params_}') # Print the best hyperparameters
		y_pred = rscv.predict(X_test) # Predict on the test set using the best model
		x_pred = rscv.predict(X_train) # Predict on training using the best model

		# Evaluate Model
		Training_F1_score, Testing_F1_score = evaluate_model([x_pred, y_train, y_pred, y_test], "f1score")
		print(f"Training F1_Score: {Training_F1_score:.2f}%")
		print(f"Testing F1_Score: {Testing_F1_score:.2f}%")

		Training_recall, Testing_recall = evaluate_model([x_pred, y_train, y_pred, y_test], "recall")
		print(f"Training Recall: {Training_recall:.2f}%")
		print(f"Testing Recall: {Testing_recall:.2f}%")
		
	else:
		model_pipeline.fit(X_train, y_train)
		# persist trained model
		save_pipeline(model_file=model_pipeline)



if __name__ == "__main__":
	"""
	Trains a model based on the configurations settings and save it in the model folder
	"""
	data = load_from_db(config.DATA, config.QUERY)
	X_train, X_test, y_train, y_test = split_data(data)
	train_model([X_train, X_test, y_train, y_test], config.OPTIMIZE)
