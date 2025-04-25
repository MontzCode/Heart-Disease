# src/data_processing.py
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import logging
from . import config # Relative import

logger = logging.getLogger(__name__) # Get logger for this module

def fetch_and_prepare_data(dataset_id: int = config.DATASET_ID,
                            target_column: str = config.TARGET_COLUMN,
                            binary_target: str = config.BINARY_TARGET_COLUMN,
                            test_split_ratio: float = config.TEST_SPLIT_RATIO,
                            random_state: int = config.RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """ Fetches, prepares (binary target), and splits data. """
    logger.info(f"Fetching dataset ID: {dataset_id}")
    try:
        dataset = fetch_ucirepo(id=dataset_id)
        X = dataset.data.features
        y = dataset.data.targets[target_column]
        logger.info("Dataset fetched.")

        df = X.copy()
        df[target_column] = y
        df.to_csv(config.RAW_DATA_FILE, index=False)
        logger.info(f"Raw data saved to {config.RAW_DATA_FILE}")

        df[binary_target] = df[target_column].apply(lambda x: 0 if x == 0 else 1)
        df = df.drop(columns=[target_column])
        logger.info(f"Created binary target column '{binary_target}'. Distribution:\n{df[binary_target].value_counts(normalize=True)}")

        if df.isnull().sum().sum() > 0:
             logger.warning(f"Missing values detected BEFORE split:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
        else:
            logger.info("No missing values detected before split.")

        X = df.drop(columns=[binary_target])
        y = df[binary_target]

        logger.info(f"Splitting data (test ratio={test_split_ratio}, random_state={random_state})")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_split_ratio,
            random_state=random_state,
            stratify=y
        )
        logger.info(f"Split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Failed to fetch or prepare data: {e}", exc_info=True) # Log traceback
        raise