# 🔬 How It Works - Detailed Explanation

## Overview

This framework implements a complete machine learning pipeline for algorithmic trading. Here's how each component works together:

```
Data → Features → Model → Signals → Backtest → Results
```

---

## 1. Data Loading & Preprocessing

### How Data is Loaded

The `DataLoader` class handles data acquisition:

```python
from utils.data_loader import DataLoader

loader = DataLoader()
data = loader.download_stock_data("AAPL", period="5y")
```

**What happens internally:**

1. **API Call**: Connects to Yahoo Finance API
2. **Download**: Retrieves OHLCV data (Open, High, Low, Close, Volume)
3. **Normalization**: Converts column names to lowercase
4. **Index Reset**: Converts date index to column
5. **Validation**: Checks for data integrity

**Data Structure:**
```
| date       | open   | high   | low    | close  | volume    |
|------------|--------|--------|--------|--------|-----------|
| 2020-01-02 | 100.50 | 102.00 | 99.80  | 101.20 | 1,500,000 |
| 2020-01-03 | 101.00 | 103.50 | 100.50 | 102.80 | 1,800,000 |
```

---

## 2. Feature Engineering

### Technical Indicators Calculation

The `TechnicalIndicators` class computes 32+ indicators:

```python
from features.technical_indicators import TechnicalIndicators

indicators = TechnicalIndicators()
data_with_features = indicators.add_all_features(data)
```

**Indicator Categories:**

#### A. Trend Indicators
Show market direction and momentum:

- **SMA (Simple Moving Average)**
  ```python
  SMA = sum(prices[-n:]) / n
  ```
  Smooths price data to identify trends

- **EMA (Exponential Moving Average)**
  ```python
  EMA_today = (Price_today * multiplier) + (EMA_yesterday * (1 - multiplier))
  multiplier = 2 / (period + 1)
  ```
  Gives more weight to recent prices

- **MACD (Moving Average Convergence Divergence)**
  ```python
  MACD = EMA(12) - EMA(26)
  Signal = EMA(MACD, 9)
  Histogram = MACD - Signal
  ```
  Shows momentum and trend changes

#### B. Momentum Indicators
Measure speed of price changes:

- **RSI (Relative Strength Index)**
  ```python
  RS = Average Gain / Average Loss
  RSI = 100 - (100 / (1 + RS))
  ```
  Values: 0-100 (>70 overbought, <30 oversold)

- **Stochastic Oscillator**
  ```python
  %K = 100 * (Current Close - Lowest Low) / (Highest High - Lowest Low)
  %D = SMA(%K, 3)
  ```
  Compares closing price to price range

#### C. Volatility Indicators
Measure price fluctuation:

- **Bollinger Bands**
  ```python
  Middle Band = SMA(20)
  Upper Band = Middle + (2 * StdDev)
  Lower Band = Middle - (2 * StdDev)
  BB Width = (Upper - Lower) / Middle
  ```
  Shows price volatility and potential reversal points

- **ATR (Average True Range)**
  ```python
  TR = max[(High - Low), |High - Close_prev|, |Low - Close_prev|]
  ATR = SMA(TR, 14)
  ```
  Measures market volatility

#### D. Volume Indicators
Analyze trading volume:

- **OBV (On-Balance Volume)**
  ```python
  if Close > Close_prev: OBV = OBV_prev + Volume
  if Close < Close_prev: OBV = OBV_prev - Volume
  if Close == Close_prev: OBV = OBV_prev
  ```
  Cumulative volume indicator

- **VWAP (Volume Weighted Average Price)**
  ```python
  VWAP = Σ(Price * Volume) / Σ(Volume)
  ```
  Average price weighted by volume

### Feature Matrix Example

After feature engineering:
```
| close  | sma_20 | sma_50 | rsi_14 | macd  | bb_upper | bb_lower | atr_14 | volume  |
|--------|--------|--------|--------|-------|----------|----------|--------|---------|
| 101.20 | 100.50 | 99.80  | 55.2   | 0.35  | 102.50   | 98.50    | 1.20   | 1.5M    |
| 102.80 | 100.80 | 99.85  | 58.7   | 0.42  | 103.00   | 98.60    | 1.25   | 1.8M    |
```

---

## 3. Target Variable Creation

### Classification Problem Setup

We convert price prediction into a 3-class classification problem:

```python
target = loader.create_target_variable(
    data,
    horizon=5,        # Look 5 days ahead
    threshold=0.01    # 1% threshold
)
```

**Target Classes:**
- `1`: BUY (future return > +1%)
- `0`: HOLD (future return between -1% and +1%)
- `-1`: SELL (future return < -1%)

**Calculation:**
```python
future_return = (Close[t+5] - Close[t]) / Close[t]

if future_return > 0.01:   target = 1   # BUY
elif future_return < -0.01: target = -1  # SELL
else:                       target = 0   # HOLD
```

**Example:**
```
Day 0: Close = $100, Close[+5] = $102 → Return = 2% → Target = 1 (BUY)
Day 1: Close = $100, Close[+5] = $99  → Return = -1% → Target = -1 (SELL)
Day 2: Close = $100, Close[+5] = $100.50 → Return = 0.5% → Target = 0 (HOLD)
```

---

## 4. Data Splitting

### Time-Series Aware Splitting

Unlike random splitting, we preserve temporal order:

```python
X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_training_data(
    data_with_features,
    test_size=0.2,      # 20% for testing
    validation_size=0.1  # 10% of training for validation
)
```

**Split Diagram:**
```
|------------ 72% Training ------------|-- 8% Val --|---- 20% Test ----|
[=================================][======][====================]
   Past Data                  →     Present  →      Future
```

**Why temporal splitting?**
- Prevents look-ahead bias
- Realistic evaluation
- Simulates real-world trading

---

## 5. Model Training

### Multiple Model Ensemble

We train three different models and combine them:

#### A. Random Forest
```python
rf_model = TradingModel(model_type='random_forest')
rf_model.fit(X_train, y_train, X_val, y_val)
```

**How it works:**
1. Creates 100 decision trees
2. Each tree trained on random subset of data
3. Each split uses random subset of features
4. Predictions averaged across all trees

**Advantages:**
- Handles non-linear relationships
- Resistant to overfitting
- Provides feature importance

#### B. XGBoost
```python
xgb_model = TradingModel(model_type='xgboost')
xgb_model.fit(X_train, y_train, X_val, y_val)
```

**How it works:**
1. Sequentially builds trees
2. Each tree corrects errors of previous trees
3. Uses gradient descent optimization
4. Early stopping prevents overfitting

**Advantages:**
- High accuracy
- Fast training
- Handles missing values

#### C. LightGBM
```python
lgb_model = TradingModel(model_type='lightgbm')
lgb_model.fit(X_train, y_train, X_val, y_val)
```

**How it works:**
1. Leaf-wise tree growth (vs. level-wise)
2. Histogram-based splitting
3. Efficient memory usage
4. Fast training on large datasets

**Advantages:**
- Very fast
- Low memory usage
- High accuracy

#### D. Ensemble Combination
```python
ensemble = EnsembleModel([rf_model, xgb_model, lgb_model])
```

**Soft Voting:**
```python
P_buy = (P_rf + P_xgb + P_lgb) / 3
P_sell = similar averaging
P_hold = similar averaging

Final Prediction = argmax([P_sell, P_hold, P_buy])
```

**Why ensemble?**
- Reduces variance
- More robust predictions
- Combines strengths of different algorithms

---

## 6. Signal Generation

### Converting Probabilities to Trading Signals

```python
predictions = ensemble.predict_proba(X_test)
signals = backtest.generate_signals_from_predictions(
    predictions,
    threshold=0.55  # Require 55% confidence
)
```

**Logic:**
```python
if P(BUY) > threshold:
    signal = 1  # BUY
elif P(SELL) > threshold:
    signal = -1  # SELL
else:
    signal = 0  # HOLD
```

**Threshold Impact:**
- **Lower threshold (0.50)**: More trades, more signals
- **Higher threshold (0.60)**: Fewer trades, higher confidence
- **Optimal**: Found through experimentation

**Example:**
```
Prediction: [P_sell=0.20, P_hold=0.25, P_buy=0.55]
Threshold: 0.55
Result: BUY signal (P_buy == threshold)

Prediction: [P_sell=0.30, P_hold=0.50, P_buy=0.20]
Threshold: 0.55
Result: HOLD signal (no prediction > threshold)
```

---

## 7. Backtesting

### Simulating Realistic Trading

```python
backtest = BacktestEngine(
    initial_capital=100000,
    commission=0.001,    # 0.1% per trade
    slippage=0.0005      # 0.05% slippage
)

results = backtest.run_backtest(test_data, signals, price_col='close')
```

**What happens during backtest:**

1. **Initialize**: Start with $100,000 cash
2. **Iterate**: For each day:
   ```
   - Check signal
   - Execute trade if signal triggers
   - Apply commission and slippage
   - Update portfolio value
   - Record equity
   ```
3. **Close**: Liquidate any open positions at end

**Trade Execution Example:**

**BUY Signal:**
```python
signal = 1  # BUY
price = $100
effective_price = $100 * (1 + 0.0005) = $100.05  # Add slippage
shares = ($100,000 * 0.999) / $100.05 = 999 shares  # Subtract commission
cash = $0
position = 999 shares
```

**SELL Signal:**
```python
signal = -1  # SELL
price = $105
effective_price = $105 * (1 - 0.0005) = $104.95  # Subtract slippage
cash = 999 * $104.95 * 0.999 = $104,745  # Subtract commission
shares = 0
position = closed
```

**Profit = $104,745 - $100,000 = $4,745 (4.75% return)**

---

## 8. Performance Metrics

### Comprehensive Evaluation

#### A. Total Return
```python
Total Return = (Final Value - Initial Value) / Initial Value
```
Example: ($110,000 - $100,000) / $100,000 = 10%

#### B. Annualized Return
```python
Annualized Return = (1 + Total Return) ^ (252 / n_days) - 1
```
Adjusts return to yearly basis (252 trading days/year)

#### C. Sharpe Ratio
```python
Sharpe = (Mean Daily Return / Std Daily Return) * sqrt(252)
```
Risk-adjusted return metric:
- **> 1.0**: Good
- **> 2.0**: Very good
- **> 3.0**: Excellent

#### D. Maximum Drawdown
```python
Drawdown = (Trough Value - Peak Value) / Peak Value
Max Drawdown = min(all drawdowns)
```
Largest peak-to-trough decline:
- Shows worst-case loss
- Important for risk management

#### E. Win Rate
```python
Win Rate = Winning Trades / Total Trades
```
Percentage of profitable trades

#### F. Profit Factor
```python
Profit Factor = Gross Profit / Gross Loss
```
- **> 1.0**: Profitable strategy
- **> 2.0**: Strong strategy
- **< 1.0**: Losing strategy

---

## 9. Complete Example Flow

### Step-by-Step Execution

```python
# 1. Load Data
data = loader.download_stock_data("AAPL", period="5y")
# Result: 1,258 days of OHLCV data

# 2. Add Features
data_with_features = indicators.add_all_features(data)
# Result: 37 features (5 original + 32 indicators)

# 3. Create Target
data_with_features['target'] = loader.create_target_variable(data_with_features)
# Result: 3-class labels (1, 0, -1)

# 4. Clean Data
data_clean = data_with_features.dropna()
# Result: 1,058 rows (after removing warm-up period)

# 5. Split Data
X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_training_data(data_clean)
# Result: 760 train, 85 val, 213 test samples

# 6. Train Models
ensemble = EnsembleModel([rf_model, xgb_model, lgb_model])
ensemble.fit(X_train, y_train, X_val, y_val)
# Result: 3 trained models combined

# 7. Generate Predictions
predictions = ensemble.predict_proba(X_test)
# Result: 213 probability vectors

# 8. Create Signals
signals = backtest.generate_signals_from_predictions(predictions, threshold=0.55)
# Result: 213 trading signals (1, 0, -1)

# 9. Run Backtest
results = backtest.run_backtest(test_data, signals, price_col='close')
# Result: Complete performance metrics

# 10. Analyze Results
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
```

**Final Output:**
```
Total Return: 8.45%
Sharpe Ratio: 1.23
Max Drawdown: -7.32%
Win Rate: 52.00%
Profit Factor: 1.67
```

---

## 10. Key Concepts

### Overfitting Prevention
- Time-series splitting (no future data leakage)
- Validation set for early stopping
- Ensemble reduces variance
- Feature selection based on importance

### Risk Management
- Commission and slippage modeling
- Position sizing (all-in or all-out)
- Maximum drawdown monitoring
- Stop-loss could be added

### Optimization Opportunities
- Hyperparameter tuning with Optuna
- Feature selection
- Signal threshold optimization
- Portfolio allocation strategies

---

## Conclusion

This framework provides a complete, production-ready pipeline for ML-based trading strategies. Each component is designed to be:
- **Modular**: Easy to swap models or indicators
- **Robust**: Handles edge cases and missing data
- **Realistic**: Models real-world trading costs
- **Extensible**: Easy to add new features

---

**Next Steps:** Check out the [examples/](../examples/) folder for working code!
