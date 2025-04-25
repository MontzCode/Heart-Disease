# tests/test_data_processing.py
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory relative to tests/ file location
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
# Need project root for config paths too if tests are run from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.data_processing import fetch_and_prepare_data

@pytest.mark.network # Requires internet access
def test_fetch_prepare_types_shapes():
    """ Test return types and basic shape consistency """
    try:
        X_train, X_test, y_train, y_test = fetch_and_prepare_data(test_split_ratio=0.3)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]
        assert X_train.shape[1] == X_test.shape[1]
    except Exception as e:
        pytest.fail(f"fetch_and_prepare_data failed: {e}")

@pytest.mark.network
def test_fetch_prepare_target_binary():
    """ Test target column is binary and original is dropped """
    try:
        X_train, _, y_train, _ = fetch_and_prepare_data()
        assert set(y_train.unique()).issubset({0, 1})
        assert config.BINARY_TARGET_COLUMN not in X_train.columns
        assert config.TARGET_COLUMN not in X_train.columns
    except Exception as e:
        pytest.fail(f"fetch_and_prepare_data target test failed: {e}")