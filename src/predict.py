import pandas as pd
from sklearn.model_selection import train_test_split

from config.core import config
from utils import load_from_db, load_pipeline

pipeline_file_name = f"{config.MODEL_VERSION}.pkl"
model = load_pipeline(model_file=pipeline_file_name)

def make_prediction(X_test: pd.DataFrame):
    """
    Make a prediction using the saved model
    """

    predictions = model.predict(X_test)
    results = {
        "predictions": predictions,
        "version": config.MODEL_VERSION,
    }
    return results


if __name__ == "__main__":

    data = load_from_db(config.DATA, config.QUERY)
    data = data.drop_duplicates(keep="first")
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[data.columns[data.columns != config.TARGET_VARIABLE]],
        data[config.TARGET_VARIABLE],
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
    )
    y_pred = make_prediction(X_test)["predictions"]
    print(y_pred)