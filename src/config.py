# src/config.py
import pathlib
import logging

# Define the project root directory
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent # Use resolve() for robustness

# Define paths relative to the project root
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
TEST_DIR = PROJECT_ROOT / "tests"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

RAW_DATA_FILE = DATA_DIR / "heart_disease_raw.csv"
# PROCESSED_DATA_FILE = DATA_DIR / "heart_disease_processed.csv" # Less needed if pipeline does it all

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Dataset fetch parameters
DATASET_ID = 45 # UCI Heart Disease dataset ID

# Target variable processing
TARGET_COLUMN = 'num'
BINARY_TARGET_COLUMN = 'target' # 0=absence, 1=presence

# Features (primarily for reference/potential selection)
FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Model training parameters
TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

# Logging configuration (basic example)
LOG_FILE = LOG_DIR / "pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE), # Log to file
        logging.StreamHandler() # Log to console/notebook output
    ]
)