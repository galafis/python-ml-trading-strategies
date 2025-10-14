"""
Simple Trading Strategy Example with Synthetic Data

This example demonstrates the full pipeline using synthetic data,
so it can run without internet connection.

1. Generate synthetic stock data
2. Feature engineering
3. Model training
4. Backtesting
5. Performance evaluation
"""

import warnings

import numpy as np
import pandas as pd

from backtesting.backtest_engine import BacktestEngine
from features.technical_indicators import TechnicalIndicators
from models.ml_models import EnsembleModel, TradingModel
from utils.data_loader import DataLoader

warnings.filterwarnings("ignore")


def generate_synthetic_stock_data(
    n_days=1000, initial_price=100, volatility=0.02, trend=0.0001
):
    """
    Generate synthetic stock price data with realistic patterns.

    Args:
        n_days: Number of trading days to generate
        initial_price: Starting price
        volatility: Daily volatility (standard deviation)
        trend: Daily trend (drift)

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)  # For reproducibility

    # Generate random returns with trend
    returns = np.random.normal(trend, volatility, n_days)

    # Calculate cumulative prices
    prices = initial_price * np.exp(np.cumsum(returns))

    # Generate OHLCV data
    data = pd.DataFrame(
        {
            "date": pd.date_range(start="2020-01-01", periods=n_days, freq="D"),
            "open": prices * (1 + np.random.uniform(-0.005, 0.005, n_days)),
            "high": prices * (1 + np.random.uniform(0, 0.01, n_days)),
            "low": prices * (1 - np.random.uniform(0, 0.01, n_days)),
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, n_days),
        }
    )

    # Ensure high >= close >= low
    data["high"] = data[["open", "high", "close"]].max(axis=1)
    data["low"] = data[["open", "low", "close"]].min(axis=1)

    return data


def main():
    print("=" * 80)
    print("ML Trading Strategy - Simple Example (Synthetic Data)")
    print("=" * 80)

    # Step 1: Generate Synthetic Data
    print("\n[1/5] Generating synthetic market data...")
    data = generate_synthetic_stock_data(n_days=1000, initial_price=100)
    print(f"Generated {len(data)} days of synthetic data")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    # Step 2: Feature Engineering
    print("\n[2/5] Engineering features...")
    indicators = TechnicalIndicators()
    data_with_features = indicators.add_all_features(data)

    # Create target variable
    loader = DataLoader()
    data_with_features["target"] = loader.create_target_variable(
        data_with_features, price_col="close", horizon=5, threshold=0.01
    )

    # Remove NaN values
    data_with_features = data_with_features.dropna()
    print(f"Created {len(data_with_features.columns) - len(data.columns)} indicators")
    print(f"Final dataset: {len(data_with_features)} rows")

    # Step 3: Prepare Training Data
    print("\n[3/5] Preparing training data...")
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_training_data(
        data_with_features, target_col="target", test_size=0.2, validation_size=0.1
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Step 4: Train Models
    print("\n[4/5] Training models...")

    # Train Random Forest (quick training for demo)
    print("  Training Random Forest...")
    rf_model = TradingModel(model_type="random_forest")
    rf_model.fit(X_train, y_train, X_val, y_val, n_estimators=50, max_depth=8)

    # Train XGBoost
    print("  Training XGBoost...")
    xgb_model = TradingModel(model_type="xgboost")
    xgb_model.fit(X_train, y_train, X_val, y_val, n_estimators=50, max_depth=5)

    # Create Ensemble
    print("  Creating ensemble model...")
    ensemble = EnsembleModel([rf_model, xgb_model])
    ensemble.fit(X_train, y_train, X_val, y_val)

    # Evaluate on validation set
    print("\n  Validation Performance:")
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    for name, model in [
        ("Random Forest", rf_model),
        ("XGBoost", xgb_model),
        ("Ensemble", ensemble),
    ]:
        val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, val_pred)
        prec = precision_score(y_val, val_pred, zero_division=0, average="weighted")
        rec = recall_score(y_val, val_pred, zero_division=0, average="weighted")
        f1 = f1_score(y_val, val_pred, zero_division=0, average="weighted")

        print(
            f"    {name:15s} - Accuracy: {acc:.3f}, "
            f"Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}"
        )

    # Feature Importance
    print("\n  Top 10 Most Important Features:")
    top_features = rf_model.get_feature_importance(top_n=10)
    for idx, row in top_features.iterrows():
        print(f"    {row['feature']:20s}: {row['importance']:.4f}")

    # Step 5: Backtesting
    print("\n[5/5] Running backtest...")

    # Generate predictions
    test_predictions = ensemble.predict_proba(X_test)

    # Convert predictions to signals
    backtest = BacktestEngine(
        initial_capital=100000, commission=0.001, slippage=0.0005
    )

    signals = backtest.generate_signals_from_predictions(
        test_predictions, threshold=0.55
    )

    # Get test data and add signals
    test_data = data_with_features.iloc[-len(X_test) :].copy()
    test_data = test_data.reset_index(drop=True)
    signals_series = pd.Series(signals).reset_index(drop=True)

    # Run backtest
    results = backtest.run_backtest(test_data, signals_series, price_col="close")

    # Display results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(f"Initial Capital:       ${backtest.initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${results.equity_curve.iloc[-1]:,.2f}")
    print(f"Total Return:          {results.total_return:,.2%}")
    print(f"Annualized Return:     {results.annualized_return:,.2%}")
    print(f"Sharpe Ratio:          {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown:          {results.max_drawdown:.2%}")
    print(f"Win Rate:              {results.win_rate:.2%}")
    print(f"Profit Factor:         {results.profit_factor:.2f}")
    print(f"Total Trades:          {results.total_trades}")
    print("=" * 80)

    # Compare with Buy & Hold
    buy_hold_return = (test_data["close"].iloc[-1] / test_data["close"].iloc[0]) - 1
    print(f"\nBuy & Hold Return:     {buy_hold_return:.2%}")
    print(
        f"Strategy Outperformance: {(results.total_return - buy_hold_return):.2%}"
    )

    print("\n✅ Strategy execution completed successfully!")
    print("\nNote: This example uses synthetic data for demonstration purposes.")
    print(
        "For real trading, use actual market data from examples/complete_strategy.py"
    )


if __name__ == "__main__":
    main()
