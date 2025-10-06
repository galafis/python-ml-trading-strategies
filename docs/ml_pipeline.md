# 🤖 ML Pipeline Documentation

## Complete ML Trading Pipeline

```mermaid
graph LR
    A[Raw Market Data] --> B[Data Loader]
    B --> C[Feature Engineering]
    C --> D[Technical Indicators]
    D --> E[ML Models]
    E --> F[Ensemble]
    F --> G[Signal Generation]
    G --> H[Backtesting]
    H --> I[Performance Metrics]
    
    style A fill:#e1f5ff
    style E fill:#fff4e1
    style I fill:#e8f5e9
```

## Detailed Feature Engineering Flow

```mermaid
flowchart TD
    Start[Market Data OHLCV] --> Load[Load Historical Data]
    Load --> Clean[Clean & Validate]
    Clean --> Calc[Calculate Indicators]
    
    Calc --> Trend[Trend Indicators]
    Calc --> Mom[Momentum Indicators]
    Calc --> Vol[Volatility Indicators]
    Calc --> Volume[Volume Indicators]
    
    Trend --> |SMA, EMA, MACD| Features[Feature Matrix]
    Mom --> |RSI, Stochastic, ROC| Features
    Vol --> |Bollinger, ATR, Keltner| Features
    Volume --> |OBV, VWAP, MFI| Features
    
    Features --> Lag[Add Lagged Features]
    Lag --> Scale[Feature Scaling]
    Scale --> Split[Train/Test Split]
    
    style Start fill:#e3f2fd
    style Features fill:#fff3e0
    style Split fill:#e8f5e9
```

## Model Training & Evaluation

```mermaid
sequenceDiagram
    participant Data as Historical Data
    participant FE as Feature Engineer
    participant Models as ML Models
    participant Ensemble as Ensemble
    participant BT as Backtester
    
    Data->>FE: Raw OHLCV
    FE->>FE: Calculate 32+ Indicators
    FE->>Models: Feature Matrix
    
    par Train Random Forest
        Models->>Models: RF Training
    and Train XGBoost
        Models->>Models: XGB Training
    and Train LightGBM
        Models->>Models: LGBM Training
    end
    
    Models->>Ensemble: Individual Predictions
    Ensemble->>Ensemble: Weighted Average
    Ensemble->>BT: Trading Signals
    BT->>BT: Simulate Trades
    BT-->>Data: Performance Metrics
```

## Feature Engineering Details

### 32+ Technical Indicators

#### Trend Indicators
- **SMA** (Simple Moving Average): 10, 20, 50, 200 periods
- **EMA** (Exponential Moving Average): 12, 26 periods
- **MACD** (Moving Average Convergence Divergence)
- **MACD Signal** & **MACD Histogram**

#### Momentum Indicators
- **RSI** (Relative Strength Index): 14 periods
- **Stochastic Oscillator**: %K and %D
- **ROC** (Rate of Change): 10 periods
- **Williams %R**: 14 periods

#### Volatility Indicators
- **Bollinger Bands**: Upper, Middle, Lower
- **Bollinger Band Width**
- **ATR** (Average True Range): 14 periods
- **Keltner Channels**: Upper, Middle, Lower

#### Volume Indicators
- **OBV** (On-Balance Volume)
- **VWAP** (Volume Weighted Average Price)
- **MFI** (Money Flow Index): 14 periods
- **Volume Rate of Change**

#### Statistical Features
- **Returns**: Daily, log returns
- **Volatility**: Rolling std (20 periods)
- **Skewness** & **Kurtosis**: Rolling 20 periods
- **Z-Score**: Price normalization

## Model Architecture

### Random Forest
```
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    random_state=42
)
```

### XGBoost
```
XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
```

### LightGBM
```
LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
```

### Ensemble Strategy
```python
ensemble_prediction = (
    0.4 * rf_prediction +
    0.3 * xgb_prediction +
    0.3 * lgbm_prediction
)
```

## Backtesting Engine

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> Analyzing: New Signal
    Analyzing --> LongPosition: Buy Signal
    Analyzing --> ShortPosition: Sell Signal
    Analyzing --> Idle: Hold Signal
    
    LongPosition --> Idle: Exit Long
    ShortPosition --> Idle: Exit Short
    
    LongPosition --> LongPosition: Hold Long
    ShortPosition --> ShortPosition: Hold Short
```

### Performance Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Total Return** | `(Final - Initial) / Initial` | Overall profit/loss |
| **Annualized Return** | `(1 + Total Return)^(252/days) - 1` | Yearly return |
| **Sharpe Ratio** | `(Return - RiskFree) / Volatility` | Risk-adjusted return |
| **Max Drawdown** | `min((Peak - Trough) / Peak)` | Largest peak-to-trough decline |
| **Win Rate** | `Winning Trades / Total Trades` | Percentage of profitable trades |
| **Profit Factor** | `Gross Profit / Gross Loss` | Profitability ratio |

## Signal Generation Logic

```python
def generate_signals(predictions, threshold=0.5):
    """
    Convert ML predictions to trading signals
    
    Args:
        predictions: Model output probabilities
        threshold: Decision threshold
    
    Returns:
        signals: 1 (Buy), -1 (Sell), 0 (Hold)
    """
    signals = np.where(predictions > threshold + 0.1, 1,  # Strong buy
              np.where(predictions < threshold - 0.1, -1, # Strong sell
              0))  # Hold
    return signals
```

## Data Flow Example

```mermaid
graph TD
    A[Yahoo Finance API] -->|Download| B[Raw OHLCV Data]
    B -->|5 years| C[AAPL Stock Data]
    C -->|Feature Engineering| D[32 Features]
    D -->|Train/Test Split| E[80% Train / 20% Test]
    E -->|Training| F[3 ML Models]
    F -->|Predictions| G[Ensemble Model]
    G -->|Signals| H[Backtest Engine]
    H -->|Metrics| I[Performance Report]
    
    I -->|Total Return| J[15.25%]
    I -->|Sharpe Ratio| K[0.99]
    I -->|Max Drawdown| L[-9.45%]
    I -->|Win Rate| M[50%]
```

## Optimization Strategies

### Hyperparameter Tuning
- Grid Search for optimal parameters
- Cross-validation (5-fold)
- Walk-forward optimization

### Feature Selection
- Correlation analysis
- Feature importance ranking
- Recursive feature elimination

### Risk Management
- Position sizing based on volatility
- Stop-loss orders
- Take-profit targets
- Maximum position limits

## Future Enhancements

1. **Deep Learning Models**
   - LSTM for sequence prediction
   - CNN for pattern recognition
   - Transformer models

2. **Alternative Data**
   - Sentiment analysis
   - News feeds
   - Social media signals

3. **Portfolio Optimization**
   - Multi-asset allocation
   - Risk parity
   - Mean-variance optimization

4. **Real-time Deployment**
   - Live data streaming
   - Low-latency prediction
   - Automated execution
