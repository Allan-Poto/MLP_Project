from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import recall_score, f1_score

from config.core import config
from pipeline import model_pipeline, tuning_pipeline
from utils import load_from_db, save_pipeline


def train_model(optimize: bool=False):
	"""
	Trains a model using the pipeline.
	If optimize is set to True, the model runs the hyperparameter tuning pipeline instead and
	prints out the following:
		1. Best paramst
		2. Training recall and f1_score
		3. Testing recall and f1_score
	"""

	# read training data
	data = load_from_db(config.DATA, config.QUERY)
	data = data.drop_duplicates(keep="first")

	# # divide train and test
	X_train, X_test, y_train, y_test = train_test_split(
		data[data.columns[data.columns != config.TARGET_VARIABLE]],
		data[config.TARGET_VARIABLE],
		test_size=config.TEST_SIZE,
		random_state=config.SEED,
	)

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
		elif config.MODEL_SELECTION == "DNN":
			pass
		else:
			raise ValueError("Model does not exist")
		rscv = RandomizedSearchCV(tuning_pipeline, param_grid, cv=config.CV, scoring="accuracy")
		rscv.fit(X_train, y_train)
		
		print(f'Best Hyperparameters: {rscv.best_params_}') # Print the best hyperparameters
		y_pred = rscv.predict(X_test) # Predict on the test set using the best model
		x_pred = rscv.predict(X_train) # Predict on training using the best model
		print(f"Training F1_Score: {f1_score(y_train, x_pred)*100:.2f}%")
		print(f"Training Recall: {recall_score(y_train, x_pred)*100:.2f}%")
		print(f"Testing F1_Score: {f1_score(y_test, y_pred)*100:.2f}%")
		print(f"Testing Recall: {recall_score(y_test, y_pred)*100:.2f}%")
		
	else:
		model_pipeline.fit(X_train, y_train)
		# persist trained model
		save_pipeline(model_file=model_pipeline)



if __name__ == "__main__":
	train_model(config.OPTIMIZE)
