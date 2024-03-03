from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score

from config.core import config
from pipeline import model_pipeline, tuning_pipeline
from utils import load_from_db, save_pipeline


def train_model(optimize: bool=False):
	"""
	Train the model
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
		rscv = RandomizedSearchCV(tuning_pipeline, config.TUNING_GRID, cv=config.CV, scoring="accuracy")
		rscv.fit(X_train, y_train)
		best_params = rscv.best_params_
		# Print the best hyperparameters
		print(f'Best Hyperparameters: {best_params}')
		# Predict on the test set using the best model
		y_pred = rscv.predict(X_test)
		# Evaluate the accuracy of the best model
		print(f"{accuracy_score(y_test, y_pred)*100:.2f}%")
		print(f"{f1_score(y_test, y_pred):.2f}")
		
	else:
		model_pipeline.fit(X_train, y_train)
		# persist trained model
		save_pipeline(model_file=model_pipeline)



if __name__ == "__main__":
	optimize = False
	#optimize = input("Please indicate True/False to optimizing: ")
	if optimize:
		train_model(optimize)
	else:
		train_model()