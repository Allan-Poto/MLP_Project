from pathlib import Path
from typing import List, Dict
from strictyaml import load
from pydantic import BaseModel

# Directory Management
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATA_DIR = PACKAGE_ROOT / "data"
MODEL_DIR = PACKAGE_ROOT / "model"

class ModelConfig(BaseModel):
    """
    All configurations related to
    Modelling
    """
    # Data Processing
    DATA: str
    QUERY: str
    TARGET_VARIABLE: str
    CATEGORICAL_FEATURES: List[str]
    NUMERICAL_FEATURES: List[str]
    DROP_FEATURES: List[str]   
    IMPUTE_MEDIAN: List[str]
    IMPUTE_MODE: List[str]
    IMPUTE_REFERENCE: List[List[str]]
    NOMINAL_CATEGORICAL: List[str]
    ORDINAL_CATEGORICAL: List[str]

    # Training Model
    MODEL_SELECTION: str
    MODEL_VERSION: str
    TEST_SIZE: float
    VALIDATION_SIZE: float
    SEED: int
    # HYPERPARAMS
    CV: int
    TUNING_GRID: Dict[str, List[float]]
    LGBM_STR_HPARAMS: Dict[str, str]
    LGBM_INT_HPARAMS: Dict[str, int]
    LGBM_FLOAT_HPARAMS: Dict[str, float]


def fetch_config_from_yaml():
    """
    Parse YAML containing the package configuration
    """
    if CONFIG_FILE_PATH.is_file():
        with open(CONFIG_FILE_PATH, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config

    raise OSError(f"Config file not found at : {CONFIG_FILE_PATH}")

config = ModelConfig(**(fetch_config_from_yaml()).data)

if __name__ == "__main__":
    print(config)