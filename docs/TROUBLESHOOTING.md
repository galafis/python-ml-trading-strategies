# 🔧 Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Issue: `ta` package installation fails
**Solution:**
```bash
# Try upgrading pip first
pip install --upgrade pip

# Then install ta
pip install ta>=0.11.0
```

#### Issue: TensorFlow installation problems
**Solution:**
TensorFlow is optional for this project. If you don't need it:
```bash
# Install without TensorFlow
pip install -r requirements.txt
# Then manually uninstall if needed
pip uninstall tensorflow
```

#### Issue: LightGBM compilation errors on Windows
**Solution:**
```bash
# Use pre-compiled wheel
pip install lightgbm --prefer-binary

# Or install from conda
conda install -c conda-forge lightgbm
```

---

### Runtime Issues

#### Issue: `ValueError: Found array with 0 sample(s)`
**Cause:** Not enough data after feature engineering and NaN removal

**Solution:**
```python
# Use longer period for data
data = loader.download_stock_data("AAPL", period="5y")  # Instead of "1y"

# Check data after feature engineering
print(f"Data shape after features: {data_with_features.shape}")
print(f"NaN counts:\n{data_with_features.isna().sum()}")

# Ensure enough data remains
data_with_features = data_with_features.dropna()
if len(data_with_features) < 100:
    raise ValueError("Not enough data points after cleaning")
```

#### Issue: Yahoo Finance connection errors
**Cause:** Network issues or rate limiting

**Solution:**
```python
# Add retry logic
import time

def download_with_retry(ticker, max_retries=3):
    for i in range(max_retries):
        try:
            data = loader.download_stock_data(ticker, period="5y")
            if len(data) > 0:
                return data
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            time.sleep(2 ** i)  # Exponential backoff
    raise ValueError("Failed to download data after retries")
```

#### Issue: Model training takes too long
**Solution:**
```python
# Reduce model complexity
rf_model = TradingModel(
    model_type='random_forest',
    n_estimators=50,  # Reduce from 100
    max_depth=5       # Reduce from 10
)

# Use smaller dataset for quick testing
X_train_small = X_train.iloc[:1000]
y_train_small = y_train.iloc[:1000]
```

#### Issue: Poor backtest performance
**Possible causes and solutions:**

1. **Overfitting:**
   ```python
   # Increase validation size
   X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_training_data(
       data_with_features,
       test_size=0.3,      # Increase test size
       validation_size=0.2  # Increase validation size
   )
   ```

2. **Wrong threshold:**
   ```python
   # Experiment with different thresholds
   for threshold in [0.50, 0.55, 0.60, 0.65]:
       signals = backtest.generate_signals_from_predictions(predictions, threshold=threshold)
       results = backtest.run_backtest(test_data, signals, price_col='close')
       print(f"Threshold {threshold}: Return = {results.total_return:.2%}")
   ```

3. **Commission/slippage too high:**
   ```python
   # Adjust costs
   backtest = BacktestEngine(
       initial_capital=100000,
       commission=0.0001,  # 0.01% instead of 0.1%
       slippage=0.0001     # 0.01% instead of 0.05%
   )
   ```

---

### Data Issues

#### Issue: Missing features after `add_all_features()`
**Solution:**
```python
# Check for required columns
required_cols = ['open', 'high', 'low', 'close', 'volume']
missing = [col for col in required_cols if col not in data.columns]
if missing:
    print(f"Missing columns: {missing}")
    
# Ensure lowercase column names
data.columns = [col.lower() for col in data.columns]
```

#### Issue: NaN values in features
**Cause:** Indicators need warm-up period

**Solution:**
```python
# Check NaN distribution
print(data_with_features.isna().sum().sort_values(ascending=False))

# Drop NaN rows
data_clean = data_with_features.dropna()
print(f"Rows before: {len(data_with_features)}, after: {len(data_clean)}")

# Or use forward fill for some features
data_with_features = data_with_features.fillna(method='ffill')
```

---

### Testing Issues

#### Issue: Tests fail with `ModuleNotFoundError`
**Solution:**
```bash
# Ensure PYTHONPATH is set correctly
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use pytest with explicit path
PYTHONPATH=$PYTHONPATH:. pytest tests/ -v
```

#### Issue: Import errors in examples
**Solution:**
```python
# Add proper path handling in scripts
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Now imports should work
from utils.data_loader import DataLoader
```

---

## Performance Optimization

### Speed up data loading
```python
# Cache downloaded data
import joblib

cache_file = f"data/raw/{ticker}_5y.pkl"
if os.path.exists(cache_file):
    data = joblib.load(cache_file)
else:
    data = loader.download_stock_data(ticker, period="5y")
    joblib.dump(data, cache_file)
```

### Speed up model training
```python
# Use parallel processing
rf_model = TradingModel(
    model_type='random_forest',
    n_jobs=-1  # Use all CPU cores
)

# Reduce feature set
top_features = rf_model.get_feature_importance(top_n=20)
important_cols = top_features['feature'].tolist()
X_train_reduced = X_train[important_cols]
```

### Memory optimization
```python
# Use float32 instead of float64
X_train = X_train.astype('float32')
y_train = y_train.astype('int32')

# Delete unused variables
del data_with_features
import gc
gc.collect()
```

---

## Debugging Tips

### Enable verbose logging
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In your code
logger.info(f"Training data shape: {X_train.shape}")
logger.info(f"Target distribution: {y_train.value_counts()}")
```

### Inspect model predictions
```python
# Check prediction distribution
predictions = model.predict_proba(X_test)
print(f"Mean prediction: {predictions[:, 1].mean():.3f}")
print(f"Std prediction: {predictions[:, 1].std():.3f}")

# Visualize predictions
import matplotlib.pyplot as plt
plt.hist(predictions[:, 1], bins=50)
plt.title('Prediction Distribution')
plt.show()
```

### Validate data quality
```python
# Check for data issues
print(f"Data types:\n{data.dtypes}")
print(f"\nValue ranges:")
for col in ['open', 'high', 'low', 'close']:
    print(f"{col}: [{data[col].min():.2f}, {data[col].max():.2f}]")

# Check for anomalies
print(f"\nZero volumes: {(data['volume'] == 0).sum()}")
print(f"Negative prices: {(data['close'] < 0).sum()}")
```

---

## Getting Help

If you're still experiencing issues:

1. **Check existing issues**: https://github.com/galafis/python-ml-trading-strategies/issues
2. **Create a new issue**: Include:
   - Python version
   - Package versions (`pip list`)
   - Full error message
   - Minimal code to reproduce
3. **Check documentation**: Review `docs/` folder for detailed guides

---

## FAQ Troubleshooting

**Q: Can I use this with cryptocurrency data?**
A: Yes, but you'll need to adapt the data loader. Use a crypto API like CoinGecko or Binance.

**Q: Why are my returns different from buy-and-hold?**
A: This is normal. ML strategies may underperform or outperform depending on market conditions.

**Q: Can I use this for live trading?**
A: This is a backtesting framework. For live trading, add real-time data feeds and order execution logic.

**Q: How do I improve model performance?**
A: Try:
- More data (longer period)
- Feature engineering (add more indicators)
- Hyperparameter tuning
- Different models
- Better risk management

---

**Need more help?** Open an issue on GitHub or check the documentation!
