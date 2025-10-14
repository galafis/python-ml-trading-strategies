# 📓 Jupyter Notebooks

This directory contains interactive Jupyter notebooks for learning and experimentation.

## 📚 Notebooks Overview

### 1. [Getting Started Tutorial](01_getting_started_tutorial.ipynb)
**Level:** Beginner  
**Duration:** 30-45 minutes

Learn the fundamentals of the ML Trading Strategies framework:
- Loading and visualizing market data
- Creating technical indicators
- Training multiple ML models
- Creating ensemble models
- Analyzing feature importance
- Running backtests
- Interpreting results

**Prerequisites:** None - starts from scratch!

### 2. [Advanced Analysis](02_advanced_analysis.ipynb)
**Level:** Intermediate to Advanced  
**Duration:** 45-60 minutes

Deep dive into advanced techniques:
- Feature correlation analysis
- Hyperparameter optimization with Optuna
- SHAP value interpretation
- Prediction confidence analysis
- Model interpretability

**Prerequisites:** Complete the getting started tutorial first

## 🚀 Quick Start

### Installation

Make sure you have Jupyter installed:

```bash
pip install jupyter notebook
# or
pip install jupyterlab
```

### Running Notebooks

From the project root directory:

```bash
# Start Jupyter Notebook
jupyter notebook notebooks/

# Or start JupyterLab
jupyter lab notebooks/
```

Then open any notebook (`.ipynb` file) in your browser.

### Setting up the Environment

All notebooks automatically add the `src` directory to the Python path, so you can import modules directly:

```python
from utils.data_loader import DataLoader
from features.technical_indicators import TechnicalIndicators
from models.ml_models import TradingModel, EnsembleModel
from backtesting.backtest_engine import BacktestEngine
```

## 📊 What You'll Learn

### Notebook 1: Getting Started
- ✅ Data loading and preparation
- ✅ Technical indicator creation
- ✅ Model training workflow
- ✅ Ensemble methods
- ✅ Backtesting strategies
- ✅ Performance visualization

### Notebook 2: Advanced Analysis
- ✅ Feature selection techniques
- ✅ Automated hyperparameter tuning
- ✅ Model interpretability with SHAP
- ✅ Confidence-based trading signals
- ✅ Advanced visualization techniques

## 🛠️ Technologies Used

- **Jupyter Notebook/Lab**: Interactive development environment
- **matplotlib/seaborn**: Data visualization
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning
- **XGBoost/LightGBM**: Gradient boosting
- **SHAP**: Model interpretation
- **Optuna**: Hyperparameter optimization

## 💡 Tips for Success

1. **Run cells in order**: Notebooks are designed to be executed sequentially
2. **Experiment freely**: Modify parameters and see what happens
3. **Read the markdown**: Important explanations are in text cells
4. **Save your work**: Use "File > Save" frequently
5. **Restart kernel if needed**: "Kernel > Restart & Clear Output" for a fresh start

## 🎯 Learning Path

**Complete Beginner?**
1. Start with `01_getting_started_tutorial.ipynb`
2. Run each cell and read the explanations
3. Experiment with different parameters
4. Move to `02_advanced_analysis.ipynb` when comfortable

**Have ML Experience?**
1. Skim through `01_getting_started_tutorial.ipynb`
2. Focus on trading-specific sections
3. Dive deep into `02_advanced_analysis.ipynb`
4. Try creating your own custom strategies

## 📖 Additional Resources

- [Project README](../README.md) - Complete project documentation
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute
- [Documentation](../docs/) - Detailed technical documentation
- [Examples](../examples/) - Standalone Python scripts

## ❓ Common Issues

**Issue: Module not found errors**
```python
# Make sure this cell runs first in each notebook:
import sys
import os
sys.path.insert(0, os.path.abspath('../src'))
```

**Issue: Plots not showing**
```python
# Add this to enable inline plots:
%matplotlib inline
```

**Issue: Kernel dies or runs out of memory**
- Reduce data size (fewer days)
- Use fewer Optuna trials
- Restart kernel and try again

## 🤝 Contributing

Have ideas for new notebooks? Found an issue? 

1. Check [existing issues](https://github.com/galafis/python-ml-trading-strategies/issues)
2. Open a new issue or pull request
3. Follow the [Contributing Guide](../CONTRIBUTING.md)

## ⚠️ Important Disclaimers

**EDUCATIONAL PURPOSE ONLY**: These notebooks are for learning and research.

**NOT FINANCIAL ADVICE**: Do not use for real trading without thorough testing and understanding of risks.

**NO WARRANTY**: Results shown are examples only. Past performance doesn't guarantee future results.

---

**Happy Learning! 🎉**

If these notebooks helped you, please consider ⭐️ starring the repository!
