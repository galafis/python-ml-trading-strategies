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


def test_calculate_rsi():
    """Test RSI calculation"""
    # Create price data with known pattern
    data = pd.Series([44, 44.5, 45, 45.5, 45, 44.5, 44, 43.5, 44, 44.5, 45, 45.5, 46, 46.5, 47])
    rsi = TechnicalIndicators.calculate_rsi(data, period=14)
    
    # RSI should be between 0 and 100
    assert rsi.dropna().min() >= 0
    assert rsi.dropna().max() <= 100
    # First period values (except the last one) should be NaN due to rolling window
    assert pd.isna(rsi.iloc[:13]).all()
    # After period, we should have valid values
    assert not pd.isna(rsi.iloc[13])


def test_calculate_macd():
    """Test MACD calculation"""
    data = pd.Series(range(1, 101))
    macd, signal, hist = TechnicalIndicators.calculate_macd(data)
    
    assert len(macd) == len(data)
    assert len(signal) == len(data)
    assert len(hist) == len(data)
    # Histogram should equal macd - signal
    np.testing.assert_array_almost_equal(
        hist.dropna().values, 
        (macd - signal).dropna().values
    )


def test_calculate_bollinger_bands():
    """Test Bollinger Bands calculation"""
    data = pd.Series([10, 12, 11, 13, 12, 14, 13, 15, 14, 16] * 3)
    upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(data, period=10, num_std=2)
    
    assert len(upper) == len(data)
    assert len(middle) == len(data)
    assert len(lower) == len(data)
    
    # Upper should be above middle, middle above lower
    valid_idx = middle.notna()
    assert (upper[valid_idx] >= middle[valid_idx]).all()
    assert (middle[valid_idx] >= lower[valid_idx]).all()


def test_calculate_atr():
    """Test ATR calculation"""
    high = pd.Series([105, 108, 107, 110, 112, 111, 115, 114, 118, 120])
    low = pd.Series([95, 97, 96, 99, 101, 100, 104, 103, 107, 109])
    close = pd.Series([100, 102, 101, 105, 107, 106, 110, 109, 113, 115])
    
    atr = TechnicalIndicators.calculate_atr(high, low, close, period=5)
    
    assert len(atr) == len(close)
    # ATR should be positive
    assert (atr.dropna() > 0).all()


def test_calculate_stochastic():
    """Test Stochastic Oscillator calculation"""
    high = pd.Series([105, 108, 107, 110, 112, 111, 115, 114, 118, 120] * 2)
    low = pd.Series([95, 97, 96, 99, 101, 100, 104, 103, 107, 109] * 2)
    close = pd.Series([100, 102, 101, 105, 107, 106, 110, 109, 113, 115] * 2)
    
    k_line, d_line = TechnicalIndicators.calculate_stochastic(high, low, close)
    
    assert len(k_line) == len(close)
    assert len(d_line) == len(close)
    # Stochastic should be between 0 and 100
    assert (k_line.dropna() >= 0).all() and (k_line.dropna() <= 100).all()
    assert (d_line.dropna() >= 0).all() and (d_line.dropna() <= 100).all()


def test_calculate_obv():
    """Test On-Balance Volume calculation"""
    close = pd.Series([100, 102, 101, 105, 103, 107])
    volume = pd.Series([1000, 1100, 900, 1200, 800, 1300])
    
    obv = TechnicalIndicators.calculate_obv(close, volume)
    
    assert len(obv) == len(close)
    # OBV is cumulative
    assert obv.iloc[0] == 0  # First value should be 0


def test_calculate_adx():
    """Test ADX calculation"""
    high = pd.Series([105, 108, 107, 110, 112, 111, 115, 114, 118, 120] * 3)
    low = pd.Series([95, 97, 96, 99, 101, 100, 104, 103, 107, 109] * 3)
    close = pd.Series([100, 102, 101, 105, 107, 106, 110, 109, 113, 115] * 3)
    
    adx = TechnicalIndicators.calculate_adx(high, low, close, period=14)
    
    assert len(adx) == len(close)
    # ADX should be between 0 and 100
    valid_adx = adx.dropna()
    assert (valid_adx >= 0).all() and (valid_adx <= 100).all()


def test_calculate_vwap():
    """Test VWAP calculation"""
    high = pd.Series([105, 108, 107, 110, 112])
    low = pd.Series([95, 97, 96, 99, 101])
    close = pd.Series([100, 102, 101, 105, 107])
    volume = pd.Series([1000, 1100, 900, 1200, 800])
    
    vwap = TechnicalIndicators.calculate_vwap(high, low, close, volume)
    
    assert len(vwap) == len(close)
    # VWAP should be within the price range
    assert (vwap >= low.min()).all()
    assert (vwap <= high.max()).all()


def test_add_all_features():
    """Test adding all technical indicators to a dataframe"""
    # Create sample OHLCV data
    data = pd.DataFrame({
        'close': np.random.uniform(90, 110, 300),
        'high': np.random.uniform(100, 120, 300),
        'low': np.random.uniform(80, 100, 300),
        'volume': np.random.uniform(1000, 2000, 300)
    })
    
    result = TechnicalIndicators.add_all_features(data)
    
    # Should have more columns than original
    assert len(result.columns) > len(data.columns)
    
    # Check for key indicators
    expected_indicators = [
        'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
        'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'atr_14', 'stoch_k', 'stoch_d', 'obv', 'adx_14', 'vwap',
        'return_1d', 'return_5d', 'return_10d', 'return_20d',
        'volatility_5d', 'volatility_10d', 'volatility_20d'
    ]
    
    for indicator in expected_indicators:
        assert indicator in result.columns, f"Missing indicator: {indicator}"
    
    # Original columns should still exist
    for col in data.columns:
        assert col in result.columns
