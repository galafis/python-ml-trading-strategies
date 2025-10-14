"""
Tests for the backtesting engine module.
"""

import pandas as pd
import numpy as np
import pytest
from src.backtesting.backtest_engine import BacktestEngine, BacktestResults


def test_backtest_engine_initialization():
    """Test backtest engine initialization"""
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005
    )
    
    assert engine.initial_capital == 100000
    assert engine.commission == 0.001
    assert engine.slippage == 0.0005


def test_generate_signals_from_predictions_binary():
    """Test signal generation from binary predictions"""
    # Binary class predictions
    predictions = np.array([0, 1, 1, 0, 1])
    signals = BacktestEngine.generate_signals_from_predictions(predictions)
    
    assert len(signals) == 5
    assert signals.iloc[0] == -1  # class 0 -> sell
    assert signals.iloc[1] == 1   # class 1 -> buy
    assert signals.iloc[2] == 1   # class 1 -> buy


def test_generate_signals_from_predictions_proba():
    """Test signal generation from probability predictions"""
    # Probability predictions [prob_class_0, prob_class_1]
    predictions = np.array([
        [0.8, 0.2],  # Low confidence -> sell
        [0.3, 0.7],  # High confidence -> buy
        [0.45, 0.55], # Medium confidence -> hold
        [0.1, 0.9],  # Very high confidence -> buy
    ])
    
    signals = BacktestEngine.generate_signals_from_predictions(predictions, threshold=0.6)
    
    assert len(signals) == 4
    assert signals.iloc[0] == -1  # prob < 0.4 -> sell
    assert signals.iloc[1] == 1   # prob > 0.6 -> buy
    assert signals.iloc[2] == 0   # 0.4 <= prob <= 0.6 -> hold
    assert signals.iloc[3] == 1   # prob > 0.6 -> buy


def test_run_backtest_simple_buy_hold():
    """Test simple buy and hold strategy"""
    # Create simple price data
    data = pd.DataFrame({
        'close': [100, 105, 110, 115, 120],
        'high': [102, 107, 112, 117, 122],
        'low': [98, 103, 108, 113, 118],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    # Buy at start, sell at end
    signals = pd.Series([1, 0, 0, 0, -1])
    
    engine = BacktestEngine(initial_capital=10000, commission=0.001, slippage=0.0005)
    results = engine.run_backtest(data, signals, price_col='close')
    
    assert isinstance(results, BacktestResults)
    assert results.total_return > 0  # Should be profitable
    assert results.total_trades == 1  # One complete trade
    assert len(results.equity_curve) == 5


def test_run_backtest_no_trades():
    """Test backtest with no trading signals"""
    data = pd.DataFrame({
        'close': [100, 105, 110, 115, 120]
    })
    
    # All hold signals
    signals = pd.Series([0, 0, 0, 0, 0])
    
    engine = BacktestEngine(initial_capital=10000)
    results = engine.run_backtest(data, signals, price_col='close')
    
    assert results.total_trades == 0
    assert results.total_return == 0  # No change in capital


def test_run_backtest_multiple_trades():
    """Test backtest with multiple trades"""
    data = pd.DataFrame({
        'close': [100, 105, 110, 105, 115, 120, 115, 125]
    })
    
    # Multiple buy/sell signals
    signals = pd.Series([1, 0, -1, 1, 0, -1, 1, -1])
    
    engine = BacktestEngine(initial_capital=10000, commission=0.001, slippage=0.0005)
    results = engine.run_backtest(data, signals, price_col='close')
    
    assert results.total_trades >= 3  # At least 3 complete trades
    assert isinstance(results.sharpe_ratio, float)
    assert isinstance(results.max_drawdown, float)


def test_run_backtest_with_loss():
    """Test backtest with losing trade"""
    data = pd.DataFrame({
        'close': [100, 95, 90, 85, 80]
    })
    
    # Buy at start, sell at end (losing trade)
    signals = pd.Series([1, 0, 0, 0, -1])
    
    engine = BacktestEngine(initial_capital=10000, commission=0.001, slippage=0.0005)
    results = engine.run_backtest(data, signals, price_col='close')
    
    assert results.total_return < 0  # Should have loss
    assert results.win_rate == 0  # No winning trades


def test_run_backtest_empty_data():
    """Test backtest with empty data raises error"""
    data = pd.DataFrame({'close': []})
    signals = pd.Series([])
    
    engine = BacktestEngine()
    
    with pytest.raises(ValueError, match="No data available"):
        engine.run_backtest(data, signals)


def test_backtest_commission_and_slippage():
    """Test that commission and slippage are properly applied"""
    data = pd.DataFrame({
        'close': [100, 110]  # 10% price increase
    })
    
    signals = pd.Series([1, -1])  # Buy then sell
    
    # High commission and slippage
    engine = BacktestEngine(initial_capital=10000, commission=0.05, slippage=0.05)
    results = engine.run_backtest(data, signals, price_col='close')
    
    # With 5% commission and 5% slippage on both sides, 
    # even with 10% price increase, we should have a loss
    assert results.total_return < 0.1  # Less than the raw price increase


def test_backtest_results_metrics():
    """Test that all backtest metrics are calculated"""
    data = pd.DataFrame({
        'close': [100, 102, 105, 103, 108, 112, 110, 115]
    })
    
    signals = pd.Series([1, 0, 0, -1, 1, 0, -1, 0])
    
    engine = BacktestEngine(initial_capital=10000)
    results = engine.run_backtest(data, signals, price_col='close')
    
    # Check all metrics exist
    assert hasattr(results, 'total_return')
    assert hasattr(results, 'annualized_return')
    assert hasattr(results, 'sharpe_ratio')
    assert hasattr(results, 'max_drawdown')
    assert hasattr(results, 'win_rate')
    assert hasattr(results, 'profit_factor')
    assert hasattr(results, 'total_trades')
    assert hasattr(results, 'equity_curve')
    assert hasattr(results, 'trades')
    
    # Check metrics are reasonable
    assert isinstance(results.total_return, float)
    assert isinstance(results.sharpe_ratio, float)
    assert results.max_drawdown <= 0  # Drawdown should be negative or zero


def test_backtest_win_rate_calculation():
    """Test win rate calculation"""
    data = pd.DataFrame({
        'close': [100, 110, 105, 115, 110, 120]
    })
    
    # Two trades: one win (100->110), one loss (105->110->105)
    signals = pd.Series([1, -1, 1, 0, -1, 0])
    
    engine = BacktestEngine(initial_capital=10000, commission=0.0, slippage=0.0)
    results = engine.run_backtest(data, signals, price_col='close')
    
    # Should have win rate between 0 and 1
    assert 0 <= results.win_rate <= 1
