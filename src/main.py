from config.core import config
from train import train_model
from utils import split_data, evaluate_model, load_from_db, load_pipeline
from predict import make_prediction

if __name__ == "__main__":
	# Data Preparation
	data = load_from_db(config.DATA, config.QUERY)
	X_train, X_test, y_train, y_test = split_data(data)

	# Train new model
	train_model([X_train, X_test, y_train, y_test], False)

	# Load Model
	pipeline_file_name = f"{config.MODEL_SELECTION}_{config.MODEL_VERSION}.pkl"
	model = load_pipeline(model_file=pipeline_file_name)

	# Make Predictions
	x_pred = make_prediction(model, X_train)["predictions"]
	y_pred = make_prediction(model, X_test)["predictions"]
	print("\n")

	# Evaluate Model
	Training_F1_score, Testing_F1_score = evaluate_model([x_pred, y_train, y_pred, y_test], "f1score")
	print(f"Training F1_Score: {Training_F1_score:.2f}%")
	print(f"Testing F1_Score: {Testing_F1_score:.2f}%")

	Training_recall, Testing_recall = evaluate_model([x_pred, y_train, y_pred, y_test], "recall")
	print(f"Training Recall: {Training_recall:.2f}%")
	print(f"Testing Recall: {Testing_recall:.2f}%")
