# 🤖 Machine Learning Trading Strategies

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green.svg)](https://xgboost.ai/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[English](#english) | [Português](#português)

---

<a name="english"></a>

## 📖 Overview

A **comprehensive machine learning framework** for developing, testing, and deploying quantitative trading strategies. This project provides end-to-end pipeline from feature engineering to backtesting, with production-ready code and extensive documentation.

### Key Features

- **📊 Advanced Feature Engineering**: 32+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **🤖 Multiple ML Models**: Random Forest, XGBoost, LightGBM, and Ensemble methods
- **📈 Comprehensive Backtesting**: Full backtesting engine with performance metrics
- **🎯 Risk-Adjusted Metrics**: Sharpe ratio, maximum drawdown, win rate, profit factor
- **🔄 Complete Pipeline**: Data loading → Feature engineering → Training → Backtesting
- **📉 Real Market Data**: Integration with Yahoo Finance for live data
- **🧪 Production-Ready**: Clean code, type hints, comprehensive documentation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  ML Trading Strategy Pipeline                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │     Data     │───▶│   Feature    │───▶│   Training   │  │
│  │   Loading    │    │ Engineering  │    │    Models    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                    │                    │          │
│         ▼                    ▼                    ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Market     │    │  Technical   │    │  Prediction  │  │
│  │    Data      │    │  Indicators  │    │   & Signals  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                   │          │
│                                                   ▼          │
│                                          ┌──────────────┐   │
│                                          │ Backtesting  │   │
│                                          │   Engine     │   │
│                                          └──────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **pip** or **conda**

### Installation

```bash
# Clone the repository
git clone https://github.com/galafis/python-ml-trading-strategies.git
cd python-ml-trading-strategies

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Running the Complete Example

```bash
cd examples
python complete_strategy.py
```

---

## 💻 Usage Examples

### 1. Data Loading

```python
from utils.data_loader import DataLoader

# Load stock data
loader = DataLoader()
data = loader.download_stock_data("AAPL", period="5y")
print(f"Loaded {len(data)} days of data")
```

### 2. Feature Engineering

```python
from features.technical_indicators import TechnicalIndicators

# Add all technical indicators
indicators = TechnicalIndicators()
data_with_features = indicators.add_all_features(data)

# Create target variable (predict 5-day returns)
data_with_features['target'] = loader.create_target_variable(
    data_with_features,
    horizon=5,
    threshold=0.01  # 1% threshold
)
```

### 3. Model Training

```python
from models.ml_models import TradingModel, EnsembleModel

# Prepare data
X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_training_data(
    data_with_features,
    target_col='target'
)

# Train Random Forest
rf_model = TradingModel(model_type='random_forest')
rf_model.fit(X_train, y_train, X_val, y_val, n_estimators=100)

# Train XGBoost
xgb_model = TradingModel(model_type='xgboost')
xgb_model.fit(X_train, y_train, X_val, y_val)

# Create Ensemble
ensemble = EnsembleModel([rf_model, xgb_model])
ensemble.fit(X_train, y_train, X_val, y_val)

# Get feature importance
print(rf_model.get_feature_importance(top_n=10))
```

### 4. Backtesting

```python
from backtesting.backtest_engine import BacktestEngine

# Initialize backtest engine
backtest = BacktestEngine(
    initial_capital=100000,
    commission=0.001,  # 0.1%
    slippage=0.0005    # 0.05%
)

# Generate trading signals
predictions = ensemble.predict_proba(X_test)
signals = backtest.generate_signals_from_predictions(predictions, threshold=0.55)

# Run backtest
results = backtest.run_backtest(test_data, signals, price_col='close')

# Display results
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Win Rate: {results.win_rate:.2%}")
```

---

## 📊 Technical Indicators

The framework includes 32+ technical indicators:

### Trend Indicators
- **SMA** (Simple Moving Average): 5, 10, 20, 50, 200 periods
- **EMA** (Exponential Moving Average): 5, 10, 20, 50, 200 periods
- **MACD** (Moving Average Convergence Divergence)
- **ADX** (Average Directional Index)

### Momentum Indicators
- **RSI** (Relative Strength Index)
- **Stochastic Oscillator** (%K, %D)
- **Price Momentum** (1, 5, 10, 20 days)

### Volatility Indicators
- **Bollinger Bands** (Upper, Middle, Lower, Width)
- **ATR** (Average True Range)
- **Historical Volatility** (5, 10, 20 days)

### Volume Indicators
- **OBV** (On-Balance Volume)
- **VWAP** (Volume Weighted Average Price)

---

## 🤖 Machine Learning Models

### Supported Models

1. **Random Forest Classifier**
   - Ensemble of decision trees
   - Robust to overfitting
   - Feature importance analysis

2. **XGBoost**
   - Gradient boosting framework
   - High performance
   - Early stopping support

3. **LightGBM**
   - Fast gradient boosting
   - Memory efficient
   - Handles large datasets

4. **Logistic Regression**
   - Linear baseline model
   - Fast training
   - Interpretable coefficients

5. **Ensemble Model**
   - Combines multiple models
   - Voting mechanism (hard/soft)
   - Improved robustness

### Model Features

- **Automatic Feature Scaling**: StandardScaler preprocessing
- **Early Stopping**: Prevents overfitting
- **Feature Importance**: Identify key predictors
- **Model Persistence**: Save/load trained models
- **Hyperparameter Optimization**: Ready for Optuna integration

---

## 📈 Backtesting Engine

### Features

- **Realistic Trading Simulation**: Commission and slippage modeling
- **Position Management**: Long-only strategies
- **Performance Metrics**:
  - Total Return
  - Annualized Return
  - Sharpe Ratio
  - Maximum Drawdown
  - Win Rate
  - Profit Factor
  - Total Trades

### Example Results

```
================================================================================
BACKTEST RESULTS
================================================================================
Initial Capital:      $100,000.00
Final Portfolio Value: $102,078.62
Total Return:         2.08%
Annualized Return:    2.48%
Sharpe Ratio:         0.22
Max Drawdown:         -16.05%
Win Rate:             50.00%
Profit Factor:        3.14
Total Trades:         4
================================================================================
```

---

## 📁 Project Structure

```
python-ml-trading-strategies/
├── src/
│   ├── features/
│   │   ├── technical_indicators.py   # Technical indicator calculations
│   │   └── __init__.py
│   ├── models/
│   │   ├── ml_models.py              # ML model implementations
│   │   └── __init__.py
│   ├── backtesting/
│   │   ├── backtest_engine.py        # Backtesting framework
│   │   └── __init__.py
│   ├── utils/
│   │   ├── data_loader.py            # Data loading utilities
│   │   └── __init__.py
│   └── __init__.py
├── examples/
│   └── complete_strategy.py          # End-to-end example
├── tests/                            # Unit tests
├── data/                             # Data storage
│   ├── raw/
│   └── processed/
├── notebooks/                        # Jupyter notebooks
├── docs/                             # Documentation
├── requirements.txt                  # Dependencies
├── setup.py                          # Package setup
└── README.md                         # This file
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| ML Framework | scikit-learn, XGBoost, LightGBM |
| Data Processing | pandas, numpy |
| Market Data | yfinance |
| Visualization | matplotlib, seaborn, plotly |
| Optimization | optuna |
| Model Interpretation | shap |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_features.py
```

---

## 📊 Performance Optimization

### Tips for Better Results

1. **Feature Selection**: Use feature importance to remove noisy features
2. **Hyperparameter Tuning**: Use Optuna for systematic optimization
3. **Ensemble Methods**: Combine multiple models for robustness
4. **Cross-Validation**: Use time-series cross-validation
5. **Risk Management**: Adjust position sizing based on confidence
6. **Transaction Costs**: Model realistic commissions and slippage

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards

- Follow PEP 8 style guide
- Add type hints to functions
- Write docstrings for classes and methods
- Add unit tests for new features
- Run `black` for code formatting
- Run `flake8` for linting

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-lafis)

---

## 🙏 Acknowledgments

- Financial markets for inspiration
- Open-source ML community
- Contributors and users

---

<a name="português"></a>

## 📖 Visão Geral

Um **framework abrangente de machine learning** para desenvolver, testar e implantar estratégias de trading quantitativo. Este projeto fornece pipeline completo desde engenharia de features até backtesting, com código pronto para produção e documentação extensiva.

### Principais Recursos

- **📊 Engenharia Avançada de Features**: Mais de 32 indicadores técnicos
- **🤖 Múltiplos Modelos ML**: Random Forest, XGBoost, LightGBM e métodos Ensemble
- **📈 Backtesting Abrangente**: Motor completo de backtesting com métricas de performance
- **🎯 Métricas Ajustadas ao Risco**: Sharpe ratio, drawdown máximo, taxa de acerto
- **🔄 Pipeline Completo**: Carregamento de dados → Engenharia de features → Treinamento → Backtesting
- **📉 Dados Reais de Mercado**: Integração com Yahoo Finance
- **🧪 Pronto para Produção**: Código limpo, type hints, documentação completa

---

## 🚀 Início Rápido

### Pré-requisitos

- **Python 3.9+**
- **pip** ou **conda**

### Instalação

```bash
# Clone o repositório
git clone https://github.com/galafis/python-ml-trading-strategies.git
cd python-ml-trading-strategies

# Instale as dependências
pip install -r requirements.txt

# Ou instale em modo de desenvolvimento
pip install -e .
```

### Executando o Exemplo Completo

```bash
cd examples
python complete_strategy.py
```

---

## 💻 Exemplos de Uso

### 1. Carregamento de Dados

```python
from utils.data_loader import DataLoader

# Carregar dados de ações
loader = DataLoader()
data = loader.download_stock_data("AAPL", period="5y")
print(f"Carregados {len(data)} dias de dados")
```

### 2. Engenharia de Features

```python
from features.technical_indicators import TechnicalIndicators

# Adicionar todos os indicadores técnicos
indicators = TechnicalIndicators()
data_with_features = indicators.add_all_features(data)

# Criar variável alvo (prever retornos de 5 dias)
data_with_features['target'] = loader.create_target_variable(
    data_with_features,
    horizon=5,
    threshold=0.01  # Threshold de 1%
)
```

### 3. Treinamento de Modelos

```python
from models.ml_models import TradingModel, EnsembleModel

# Preparar dados
X_train, X_val, X_test, y_train, y_val, y_test = loader.prepare_training_data(
    data_with_features,
    target_col='target'
)

# Treinar Random Forest
rf_model = TradingModel(model_type='random_forest')
rf_model.fit(X_train, y_train, X_val, y_val, n_estimators=100)

# Treinar XGBoost
xgb_model = TradingModel(model_type='xgboost')
xgb_model.fit(X_train, y_train, X_val, y_val)

# Criar Ensemble
ensemble = EnsembleModel([rf_model, xgb_model])
ensemble.fit(X_train, y_train, X_val, y_val)

# Obter importância das features
print(rf_model.get_feature_importance(top_n=10))
```

### 4. Backtesting

```python
from backtesting.backtest_engine import BacktestEngine

# Inicializar motor de backtesting
backtest = BacktestEngine(
    initial_capital=100000,
    commission=0.001,  # 0.1%
    slippage=0.0005    # 0.05%
)

# Gerar sinais de trading
predictions = ensemble.predict_proba(X_test)
signals = backtest.generate_signals_from_predictions(predictions, threshold=0.55)

# Executar backtest
results = backtest.run_backtest(test_data, signals, price_col='close')

# Exibir resultados
print(f"Retorno Total: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Drawdown Máximo: {results.max_drawdown:.2%}")
print(f"Taxa de Acerto: {results.win_rate:.2%}")
```

---

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 👤 Autor

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-lafis)

---

## ⭐ Mostre seu apoio

Se este projeto foi útil para você, considere dar uma ⭐️!
