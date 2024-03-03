import numpy as np
from sklearn.model_selection import train_test_split

from config.core import config
from pipeline import model_pipeline
from utils import load_from_db, save_pipeline


def train_model():
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
    model_pipeline.fit(X_train, y_train)

    # # # persist trained model
    save_pipeline(model_file=model_pipeline)


if __name__ == "__main__":
    train_model()