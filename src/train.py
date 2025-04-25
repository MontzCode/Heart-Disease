# src/train.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump
import logging
from . import config
from .pipeline import preprocessing_pipeline

logger = logging.getLogger(__name__)

def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                model_name: str = "logistic_regression") -> Pipeline:
    """ Trains model using preprocessing pipeline, saves it. """
    logger.info(f"Starting model training: {model_name}")

    if model_name == "logistic_regression":
        model = LogisticRegression(
            solver='liblinear',
            random_state=config.RANDOM_STATE,
            class_weight='balanced',
            max_iter=1000
        )
    # Add other models here...
    else:
        logger.error(f"Model '{model_name}' not recognized.")
        raise ValueError(f"Model '{model_name}' not recognized.")

    full_pipeline = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('classifier', model)
    ])

    logger.info("Fitting the full pipeline...")
    try:
        full_pipeline.fit(X_train, y_train)
        logger.info("Pipeline training completed.")
    except Exception as e:
        logger.error(f"Error during pipeline fitting: {e}", exc_info=True)
        raise

    save_path = config.MODEL_OUTPUT_DIR / f"{model_name}_pipeline.joblib"
    try:
        dump(full_pipeline, save_path)
        logger.info(f"Trained pipeline saved to: {save_path}")
    except Exception as e:
        logger.error(f"Error saving pipeline: {e}", exc_info=True)
        # Decide whether to raise

    return full_pipeline