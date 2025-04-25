# src/evaluate.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import logging
from joblib import load
from . import config

logger = logging.getLogger(__name__)

# Helper to save plots
def _save_plot(figure, filename):
    try:
        path = config.LOG_DIR / filename
        figure.savefig(path, bbox_inches='tight')
        logger.info(f"Plot saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save plot {filename}: {e}", exc_info=True)

def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                   X_train: pd.DataFrame = None, plot_prefix: str = "eval") -> dict:
    """ Evaluates pipeline, generates plots, calculates SHAP. """
    logger.info("Starting model evaluation...")
    try:
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else pipeline.decision_function(X_test)
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise

    # --- Metrics ---
    try:
        accuracy = accuracy_score(y_test, y_pred)
        # Use zero_division=0 to avoid warnings and return 0 if denominator is 0
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "roc_auc": roc_auc}
        logger.info(f"Evaluation Metrics:\n{pd.Series(metrics)}")
        logger.info("Classification Report:\n" + classification_report(y_test, y_pred, zero_division=0))

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}", exc_info=True)
        return {"error": str(e)} # Return early if metrics fail

    # --- Plots ---
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'], ax=ax_cm)
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    ax_cm.set_title('Confusion Matrix')
    _save_plot(fig_cm, f"{plot_prefix}_confusion_matrix.png")
    plt.show() # Show inline in Jupyter

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax_roc.legend(loc="lower right")
    _save_plot(fig_roc, f"{plot_prefix}_roc_curve.png")
    plt.show() # Show inline in Jupyter

    # --- SHAP Interpretation ---
    if X_train is not None:
        logger.info("Generating SHAP explanations...")
        try:
            preprocessor = pipeline.named_steps['preprocessing']
            model = pipeline.named_steps['classifier']
            X_train_processed = preprocessor.transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            try:
                feature_names = preprocessor.get_feature_names_out(input_features=X_train.columns)
            except AttributeError:
                feature_names = X_train.columns.tolist()

            # Choose appropriate SHAP explainer
            if isinstance(model, LogisticRegression):
                explainer = shap.LinearExplainer(model, X_train_processed)
                shap_values = explainer.shap_values(X_test_processed) # Single array for Linear
            else: # Fallback (e.g., for tree models or Kernel)
                logger.warning("Using KernelExplainer (may be slow)... adapt if using Tree models.")
                background_data = shap.sample(X_train_processed, 100)
                explainer = shap.KernelExplainer(model.predict_proba, background_data)
                shap_values = explainer.shap_values(X_test_processed) # List of arrays [shap_0, shap_1]

            shap_values_for_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
            X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)

            # SHAP Summary Plot (Bar)
            fig_shap_bar, ax_shap_bar = plt.subplots()
            shap.summary_plot(shap_values_for_plot, X_test_processed_df, plot_type="bar", show=False, plot_size=None)
            ax_shap_bar.set_title("SHAP Feature Importance")
            _save_plot(plt.gcf(), f"{plot_prefix}_shap_summary_bar.png") # Use plt.gcf() to get current figure
            plt.show()

            # SHAP Summary Plot (Dot)
            fig_shap_dot, ax_shap_dot = plt.subplots()
            shap.summary_plot(shap_values_for_plot, X_test_processed_df, show=False, plot_size=None)
            ax_shap_dot.set_title("SHAP Summary Plot")
            _save_plot(plt.gcf(), f"{plot_prefix}_shap_summary_dot.png")
            plt.show()

            # Dependence Plots (Example for top feature) - Suppress plot display here
            top_feature_idx = np.argsort(np.abs(shap_values_for_plot).mean(0))[-1]
            top_feature_name = feature_names[top_feature_idx]
            fig_dep, ax_dep = plt.subplots() # Create explicit figure/axes
            shap.dependence_plot(top_feature_name, shap_values_for_plot, X_test_processed_df, ax=ax_dep, show=False)
            ax_dep.set_title(f"SHAP Dependence Plot for {top_feature_name}")
            _save_plot(fig_dep, f"{plot_prefix}_shap_dependence_{top_feature_name}.png")
            plt.close(fig_dep) # Close figure to prevent double display if show=True later
            logger.info("SHAP plots generated and saved.")


        except Exception as e:
            logger.error(f"Error during SHAP generation: {e}", exc_info=True)
    else:
        logger.warning("X_train not provided to evaluate_model. Skipping SHAP analysis.")

    return metrics