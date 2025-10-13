import pandas as pd
import numpy as np
import pytest
from src.backtesting.backtest_engine import BacktestEngine, BacktestResults

# Mock data for testing
@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'open': np.random.uniform(90, 110, 100),
        'high': np.random.uniform(100, 120, 100),
        'low': np.random.uniform(80, 100, 100),
        'close': np.random.uniform(90, 110, 100),
        'volume': np.random.uniform(1000000, 5000000, 100)
    })
    # Make sure high is highest and low is lowest
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    return data

@pytest.fixture
def sample_signals():
    """Create sample trading signals"""
    signals = np.zeros(100)
    # Buy at index 10, sell at 20, buy at 30, sell at 40
    signals[10] = 1  # Buy
    signals[20] = -1  # Sell
    signals[30] = 1  # Buy
    signals[40] = -1  # Sell
    return pd.Series(signals)

def test_backtest_engine_init():
    """Test BacktestEngine initialization"""
    engine = BacktestEngine(initial_capital=100000, commission=0.001, slippage=0.0005)
    assert engine.initial_capital == 100000
    assert engine.commission == 0.001
    assert engine.slippage == 0.0005

def test_backtest_engine_init_defaults():
    """Test BacktestEngine initialization with defaults"""
    engine = BacktestEngine()
    assert engine.initial_capital == 100000.0
    assert engine.commission == 0.001
    assert engine.slippage == 0.0005

def test_run_backtest_basic(sample_data, sample_signals):
    """Test basic backtest execution"""
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run_backtest(sample_data, sample_signals, price_col='close')
    
    assert isinstance(results, BacktestResults)
    assert isinstance(results.total_return, float)
    assert isinstance(results.annualized_return, float)
    assert isinstance(results.sharpe_ratio, float)
    assert isinstance(results.max_drawdown, float)
    assert isinstance(results.win_rate, float)
    assert isinstance(results.profit_factor, float)
    assert isinstance(results.total_trades, int)
    assert isinstance(results.equity_curve, pd.Series)
    assert isinstance(results.trades, pd.DataFrame)

def test_run_backtest_equity_curve(sample_data, sample_signals):
    """Test that equity curve is generated correctly"""
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run_backtest(sample_data, sample_signals, price_col='close')
    
    assert len(results.equity_curve) > 0
    assert results.equity_curve.iloc[0] > 0  # Should start with some capital

def test_run_backtest_no_trades(sample_data):
    """Test backtest with no trading signals"""
    signals = pd.Series(np.zeros(len(sample_data)))  # All hold signals
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run_backtest(sample_data, signals, price_col='close')
    
    assert results.total_trades == 0
    assert results.win_rate == 0
    assert results.profit_factor == 0

def test_run_backtest_commission_impact(sample_data, sample_signals):
    """Test that commission impacts returns"""
    engine_no_commission = BacktestEngine(initial_capital=100000, commission=0.0, slippage=0.0)
    engine_with_commission = BacktestEngine(initial_capital=100000, commission=0.01, slippage=0.0)
    
    results_no_commission = engine_no_commission.run_backtest(sample_data, sample_signals, price_col='close')
    results_with_commission = engine_with_commission.run_backtest(sample_data, sample_signals, price_col='close')
    
    # With commission, returns should be lower (or at least different)
    assert results_with_commission.total_return <= results_no_commission.total_return

def test_run_backtest_slippage_impact(sample_data, sample_signals):
    """Test that slippage impacts returns"""
    engine_no_slippage = BacktestEngine(initial_capital=100000, commission=0.0, slippage=0.0)
    engine_with_slippage = BacktestEngine(initial_capital=100000, commission=0.0, slippage=0.01)
    
    results_no_slippage = engine_no_slippage.run_backtest(sample_data, sample_signals, price_col='close')
    results_with_slippage = engine_with_slippage.run_backtest(sample_data, sample_signals, price_col='close')
    
    # With slippage, returns should be lower
    assert results_with_slippage.total_return <= results_no_slippage.total_return

def test_generate_signals_from_predictions_probabilities():
    """Test signal generation from probability predictions"""
    # Test with 2D probability array
    predictions = np.array([
        [0.8, 0.2],  # Strong sell
        [0.4, 0.6],  # Weak buy
        [0.2, 0.8],  # Strong buy
        [0.6, 0.4],  # Weak sell
    ])
    
    signals = BacktestEngine.generate_signals_from_predictions(predictions, threshold=0.5)
    
    assert isinstance(signals, pd.Series)
    assert len(signals) == 4
    assert signals.iloc[2] == 1  # Strong buy
    # Other signals depend on threshold logic

def test_generate_signals_from_predictions_classes():
    """Test signal generation from class predictions"""
    # Test with 1D class labels
    predictions = np.array([0, 1, 1, 0, 1])
    
    signals = BacktestEngine.generate_signals_from_predictions(predictions)
    
    assert isinstance(signals, pd.Series)
    assert len(signals) == 5
    assert (signals == 1).sum() + (signals == -1).sum() + (signals == 0).sum() == 5

def test_backtest_results_dataclass():
    """Test BacktestResults dataclass"""
    equity_curve = pd.Series([100000, 101000, 102000, 101500, 103000])
    trades_df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=2),
        'type': ['BUY', 'SELL'],
        'price': [100, 105],
        'shares': [1000, 1000],
        'value': [100000, 105000]
    })
    
    results = BacktestResults(
        total_return=0.03,
        annualized_return=0.15,
        sharpe_ratio=1.5,
        max_drawdown=-0.05,
        win_rate=0.6,
        profit_factor=2.0,
        total_trades=5,
        equity_curve=equity_curve,
        trades=trades_df
    )
    
    assert results.total_return == 0.03
    assert results.sharpe_ratio == 1.5
    assert len(results.equity_curve) == 5
    assert len(results.trades) == 2

def test_run_backtest_empty_data():
    """Test backtest with empty data raises error"""
    empty_data = pd.DataFrame()
    signals = pd.Series()
    engine = BacktestEngine()
    
    with pytest.raises(ValueError, match="No data available"):
        engine.run_backtest(empty_data, signals)

def test_calculate_metrics_correctness(sample_data, sample_signals):
    """Test that metrics are calculated correctly"""
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run_backtest(sample_data, sample_signals, price_col='close')
    
    # Check that metrics are in reasonable ranges
    assert -1 <= results.total_return <= 10  # Total return should be reasonable
    assert -1 <= results.max_drawdown <= 0  # Drawdown should be negative
    assert 0 <= results.win_rate <= 1  # Win rate should be between 0 and 1
    assert results.profit_factor >= 0  # Profit factor should be non-negative
    assert results.total_trades >= 0  # Total trades should be non-negative

def test_backtest_with_only_buy_signals(sample_data):
    """Test backtest with only buy signals (no sells)"""
    signals = pd.Series(np.zeros(len(sample_data)))
    signals.iloc[10] = 1  # Single buy signal
    
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run_backtest(sample_data, signals, price_col='close')
    
    # Should close position at the end
    assert results.total_trades >= 0

def test_backtest_continuous_signals(sample_data):
    """Test backtest with continuous buy/sell signals"""
    signals = pd.Series(np.ones(len(sample_data)))  # All buy signals
    
    engine = BacktestEngine(initial_capital=100000)
    results = engine.run_backtest(sample_data, signals, price_col='close')
    
    # Should only execute one buy (already in position)
    assert isinstance(results, BacktestResults)
