import pandas as pd
import numpy as np
from src.features.technical_indicators import TechnicalIndicators

def test_calculate_sma():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sma_3 = TechnicalIndicators.calculate_sma(data, period=3)
    expected_sma_3 = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    pd.testing.assert_series_equal(sma_3, expected_sma_3, check_dtype=False)

def test_calculate_ema():
    data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ema_3 = TechnicalIndicators.calculate_ema(data, period=3)
    # Expected values calculated manually or using a known library
    expected_ema_3 = pd.Series([
        1.0, 1.5, 2.25, 3.125, 4.0625, 5.03125, 6.015625, 7.0078125, 8.00390625, 9.001953125
    ])
    pd.testing.assert_series_equal(ema_3.round(4), expected_ema_3.round(4), check_dtype=False)

