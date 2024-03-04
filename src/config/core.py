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
    NUMERICAL: List[str]
    NOMINAL_CATEGORICAL: List[str]
    ORDINAL_CATEGORICAL: List[str]

    # Training Model
    MODEL_LIST: List[str]
    MODEL_SELECTION: str
    MODEL_VERSION: str
    TEST_SIZE: float
    VALIDATION_SIZE: float
    SEED: int
    # HYPERPARAMS
    OPTIMIZE: bool
    CV: int

    # LGBM_MODEL
#   TUNING_STR_GRID: Dict[str, List[str]]
    LGBM_TUNING_INT_GRID: Dict[str, List[int]]
    LGBM_TUNING_FLOAT_GRID: Dict[str, List[float]]
    LGBM_STR_HPARAMS: Dict[str, str]
    LGBM_INT_HPARAMS: Dict[str, int]
    LGBM_FLOAT_HPARAMS: Dict[str, float]

    # SVC_MODEL
    SVC_TUNING_STR_GRID: Dict[str, List[str]]
    SVC_TUNING_FLOAT_GRID: Dict[str, List[float]]
    SVC_STR_HPARAMS: Dict[str, str]
    SVC_FLOAT_HPARAMS: Dict[str, float]

    # MLP_MODEL
    MLP_TUNING_STR_GRID: Dict[str, List[str]]
    MLP_TUNING_INT_GRID: Dict[str, List[int]]
    MLP_TUNING_FLOAT_GRID: Dict[str, List[float]]
    MLP_STR_HPARAMS: Dict[str, str]
    MLP_INT_HPARAMS: Dict[str, int]
    MLP_FLOAT_HPARAMS: Dict[str, float]

    # LOG_REGRESSION_MODEL
    LOG_TUNING_STR_GRID: Dict[str, List[str]]
    LOG_TUNING_INT_GRID: Dict[str, List[int]]
    LOG_TUNING_FLOAT_GRID: Dict[str, List[float]]
    LOG_STR_HPARAMS: Dict[str, str]
    LOG_INT_HPARAMS: Dict[str, int]
    LOG_FLOAT_HPARAMS: Dict[str, float]


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