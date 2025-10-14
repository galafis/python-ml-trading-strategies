"""
Complete Trading Strategy Example

This example demonstrates the full pipeline:
1. Data loading
2. Feature engineering
3. Model training
4. Backtesting
5. Performance evaluation
"""

import sys
import warnings

# Add parent directory to path for imports (example usage)
sys.path.append("../src")  # noqa: E402

import pandas as pd  # noqa: E402

from backtesting.backtest_engine import BacktestEngine  # noqa: E402
from features.technical_indicators import TechnicalIndicators  # noqa: E402
from models.ml_models import EnsembleModel, TradingModel  # noqa: E402
from utils.data_loader import DataLoader  # noqa: E402

warnings.filterwarnings("ignore")


def main():
    print("=" * 80)
    print("ML Trading Strategy - Complete Example")
    print("=" * 80)

    # Step 1: Load Data
    print("\n[1/5] Loading market data...")
    loader = DataLoader()

    # Download data for a stock (e.g., AAPL)
    ticker = "AAPL"
    data = loader.download_stock_data(ticker, period="5y")
    print(f"Loaded {len(data)} days of data for {ticker}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")

    # Step 2: Feature Engineering
    print("\n[2/5] Engineering features...")
    indicators = TechnicalIndicators()
    data_with_features = indicators.add_all_features(data)

    # Create target variable (predict if price will go up in next 5 days)
    data_with_features["target"] = loader.create_target_variable(
        data_with_features, price_col="close", horizon=5, threshold=0.01  # 1% threshold
    )

    # Remove NaN values
    data_with_features = data_with_features.dropna()
    print(
        f"Created {len(data_with_features.columns) - len(data.columns)} technical indicators"
    )
    print(f"Final dataset: {len(data_with_features)} rows")

    # Step 3: Prepare Training Data
    print("\n[3/5] Preparing training data...")
    X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_training_data(
        data_with_features, target_col="target", test_size=0.2, validation_size=0.1
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Features: {len(X_train.columns)}")

    # Class distribution
    print("\nTarget distribution (train):")
    print(f"  Up (1): {(y_train == 1).sum()} ({(y_train == 1).mean():.1%})")
    print(f"  Down (0): {(y_train == 0).sum()} ({(y_train == 0).mean():.1%})")

    # Step 4: Train Models
    print("\n[4/5] Training models...")

    # Train Random Forest
    print("  Training Random Forest...")
    rf_model = TradingModel(model_type="random_forest")
    rf_model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
    )

    # Train XGBoost
    print("  Training XGBoost...")
    xgb_model = TradingModel(model_type="xgboost")
    xgb_model.fit(
        X_train, y_train, X_val, y_val, n_estimators=100, max_depth=6, learning_rate=0.1
    )

    # Train LightGBM
    print("  Training LightGBM...")
    lgb_model = TradingModel(model_type="lightgbm")
    lgb_model.fit(
        X_train, y_train, X_val, y_val, n_estimators=100, max_depth=6, learning_rate=0.1
    )

    # Create Ensemble
    print("  Creating ensemble model...")
    ensemble = EnsembleModel([rf_model, xgb_model, lgb_model])
    ensemble.fit(X_train, y_train, X_val, y_val)

    # Evaluate on validation set
    print("\n  Validation Performance:")
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)

    for name, model in [
        ("Random Forest", rf_model),
        ("XGBoost", xgb_model),
        ("LightGBM", lgb_model),
        ("Ensemble", ensemble),
    ]:
        val_pred = model.predict(X_val)
        acc = accuracy_score(y_val, val_pred)
        prec = precision_score(y_val, val_pred, zero_division=0)
        rec = recall_score(y_val, val_pred, zero_division=0)
        f1 = f1_score(y_val, val_pred, zero_division=0)

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
    backtest = BacktestEngine(initial_capital=100000, commission=0.001, slippage=0.0005)

    signals = backtest.generate_signals_from_predictions(
        test_predictions, threshold=0.55  # Require 55% confidence for buy signal
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
    print(f"Initial Capital:      ${backtest.initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${results.equity_curve.iloc[-1]:,.2f}")
    print(f"Total Return:         {results.total_return:,.2%}")
    print(f"Annualized Return:    {results.annualized_return:,.2%}")
    print(f"Sharpe Ratio:         {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown:         {results.max_drawdown:.2%}")
    print(f"Win Rate:             {results.win_rate:.2%}")
    print(f"Profit Factor:        {results.profit_factor:.2f}")
    print(f"Total Trades:         {results.total_trades}")
    print("=" * 80)

    # Compare with Buy & Hold
    buy_hold_return = (test_data["close"].iloc[-1] / test_data["close"].iloc[0]) - 1
    print(f"\nBuy & Hold Return:    {buy_hold_return:.2%}")
    print(f"Strategy Outperformance: {(results.total_return - buy_hold_return):.2%}")

    print("\n✅ Strategy execution completed successfully!")


if __name__ == "__main__":
    main()
