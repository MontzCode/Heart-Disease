# main.py
import argparse
import logging
import sys
import os
from joblib import load

# Ensure src is in path if running script directly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src import config # Setup logging
from src.data_processing import fetch_and_prepare_data
from src.train import train_model
from src.evaluate import evaluate_model

logger = logging.getLogger(__name__)

def main(model_type: str = "logistic_regression", skip_train: bool = False):
    """ Main function to orchestrate the ML pipeline """
    logger.info(f"Starting pipeline run for model: {model_type}")

    # 1. Fetch and Prepare Data
    try:
        X_train, X_test, y_train, y_test = fetch_and_prepare_data()
    except Exception as e:
        logger.error("Pipeline failed at data preparation.", exc_info=True)
        return

    # 2. Train Model (or load existing)
    pipeline_path = config.MODEL_OUTPUT_DIR / f"{model_type}_pipeline.joblib"
    if not skip_train or not pipeline_path.exists():
        if skip_train:
            logger.warning(f"Skip train specified, but model not found at {pipeline_path}. Training...")
        try:
            trained_pipeline = train_model(X_train, y_train, model_name=model_type)
        except Exception as e:
            logger.error("Pipeline failed at model training.", exc_info=True)
            return
    else:
        try:
            logger.info(f"Skipping training. Loading model from {pipeline_path}")
            trained_pipeline = load(pipeline_path)
        except Exception as e:
            logger.error(f"Failed to load existing model: {e}", exc_info=True)
            return

    # 3. Evaluate Model
    try:
        logger.info("Evaluating the model...")
        metrics = evaluate_model(trained_pipeline, X_test, y_test, X_train, plot_prefix=f"main_{model_type}")
        logger.info(f"Final Evaluation Metrics:\n{pd.Series(metrics)}")
    except Exception as e:
        logger.error("Pipeline failed during model evaluation.", exc_info=True)

    logger.info("Pipeline finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Heart Disease Classification Pipeline")
    parser.add_argument("--model", type=str, default="logistic_regression", help="Model type (e.g., logistic_regression)")
    parser.add_argument("--skip-train", action='store_true', help="Skip training and load existing model if available")
    args = parser.parse_args()
    main(model_type=args.model, skip_train=args.skip_train)