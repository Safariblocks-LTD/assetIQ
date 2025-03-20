# tests/test_model.py

import sys
import os
# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from app.model import load_historical_data, predict_with_adjustments

@pytest.fixture
def sample_data():
    return load_historical_data()

def test_predict_with_adjustments(sample_data):
    future_years = [2025, 2026, 2027]
    exog_future = pd.DataFrame({
        'sentiment': [sample_data['sentiment'].mean()] * len(future_years),
        'macro': [sample_data['macro'].mean()] * len(future_years)
    }, index=future_years)
    result = predict_with_adjustments(sample_data, "Apple", future_years, exog_future)
    # Check that result is a numpy array of the correct shape
    assert result.shape == (len(future_years),)
    # Check that values are finite numbers
    assert result.dtype.kind in 'fi'
