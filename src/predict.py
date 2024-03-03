import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score

from config.core import config
from utils import load_from_db, load_pipeline

pipeline_file_name = f"{config.MODEL_SELECTION}_{config.MODEL_VERSION}.pkl"
model = load_pipeline(model_file=pipeline_file_name)

def make_prediction(X_test: pd.DataFrame):
    """
    Make a prediction using the saved model
    """

    predictions = model.predict(X_test)
    results = {
        "predictions": predictions,
        "model": config.MODEL_SELECTION,
        "version": config.MODEL_VERSION,
    }
    return results


if __name__ == "__main__":

    # Load Data
    data = load_from_db(config.DATA, config.QUERY)
    data = data.drop_duplicates(keep="first")

    # Divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[data.columns[data.columns != config.TARGET_VARIABLE]],
        data[config.TARGET_VARIABLE],
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
    )
    y_pred = make_prediction(X_test)["predictions"]
    print(f"ACCURACY: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"F1_SCORE: {f1_score(y_test, y_pred):.2f}")
    print(f"RECALL: {recall_score(y_test, y_pred):.2f}")