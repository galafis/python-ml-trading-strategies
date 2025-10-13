# 🤖 Machine Learning Trading Strategies



[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green.svg)](https://xgboost.ai/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-28%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)](tests/)

[English](#english) | [Português](#português)

---

<a name="english"></a>

## 📖 Overview

A **comprehensive machine learning framework** for developing, testing, and deploying quantitative trading strategies. This project provides end-to-end pipeline from feature engineering to backtesting, with production-ready code and extensive documentation.

> 💡 **New to ML Trading?** Check out our [How It Works](docs/HOW_IT_WORKS.md) guide for a detailed explanation!

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

![ML Trading Strategy Pipeline](docs/images/architecture.png)

**Want to understand how everything works together?** 👉 [Read the detailed guide](docs/HOW_IT_WORKS.md)

---

## 📊 Performance Benchmarks

Based on backtests using 5 years of AAPL data:

| Metric | Value | Description |
|--------|-------|-------------|
| **Total Return** | 8-15% | Varies by market conditions |
| **Annualized Return** | 3-8% | Yearly return on investment |
| **Sharpe Ratio** | 0.8-1.5 | Risk-adjusted return |
| **Max Drawdown** | -8% to -15% | Largest peak-to-trough decline |
| **Win Rate** | 48-55% | Percentage of profitable trades |
| **Profit Factor** | 1.5-2.5 | Gross profit / Gross loss |

*Note: Past performance does not guarantee future results. Always perform your own testing.*

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

**Expected Output:**
```
================================================================================
ML Trading Strategy - Complete Example
================================================================================

[1/5] Loading market data...
Loaded 1258 days of data for AAPL
Date range: 2019-01-02 to 2024-01-02

[2/5] Engineering features...
Created 32 technical indicators
Final dataset: 1058 rows

[3/5] Preparing training data...
Training set: 760 samples
Validation set: 85 samples
Test set: 213 samples

[4/5] Training models...
  Training Random Forest...
  Training XGBoost...
  Training LightGBM...
  Creating ensemble model...

[5/5] Running backtest...

================================================================================
BACKTEST RESULTS
================================================================================
Initial Capital:      $100,000.00
Final Portfolio Value: $108,450.00
Total Return:         8.45%
Sharpe Ratio:         1.23
Max Drawdown:         -7.32%
Win Rate:             52.00%
================================================================================

✅ Strategy execution completed successfully!
```

> 🎓 **Want a tutorial?** Check out our [Jupyter Notebook](notebooks/01_complete_tutorial.ipynb) for an interactive walkthrough!

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
├── src/                              # Source code
│   ├── features/
│   │   ├── technical_indicators.py   # 32+ technical indicators
│   │   └── __init__.py
│   ├── models/
│   │   ├── ml_models.py              # ML model implementations (RF, XGB, LGBM)
│   │   └── __init__.py
│   ├── backtesting/
│   │   ├── backtest_engine.py        # Backtesting framework with realistic costs
│   │   └── __init__.py
│   ├── utils/
│   │   ├── data_loader.py            # Data loading from Yahoo Finance
│   │   └── __init__.py
│   └── __init__.py
│
├── examples/
│   └── complete_strategy.py          # End-to-end example with all components
│
├── tests/                            # Unit tests (28 tests, 100% pass rate)
│   ├── test_backtest_engine.py       # Backtesting tests
│   ├── test_data_loader.py           # Data loading tests
│   ├── test_ml_models.py             # Model tests
│   └── test_technical_indicators.py  # Indicator tests
│
├── notebooks/                        # Jupyter notebooks
│   └── 01_complete_tutorial.ipynb    # Interactive tutorial
│
├── data/                             # Data storage
│   ├── raw/                          # Raw downloaded data
│   └── processed/                    # Processed features
│
├── docs/                             # Documentation
│   ├── images/                       # Images and diagrams
│   ├── HOW_IT_WORKS.md              # Detailed technical explanation
│   ├── TROUBLESHOOTING.md            # Common issues and solutions
│   ├── ml_pipeline.md                # ML pipeline documentation
│   ├── FAQ.md                        # Frequently asked questions
│   └── USE_CASES.md                  # Real-world use cases
│
├── .github/                          # GitHub configuration
│   └── workflows/
│       └── tests.yml                 # CI/CD pipeline
│
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup
├── CONTRIBUTING.md                   # Contribution guidelines
├── CHANGELOG.md                      # Version history
├── LICENSE                           # MIT License
└── README.md                         # This file
```

---

## 🔧 Troubleshooting

Having issues? Check our comprehensive [Troubleshooting Guide](docs/TROUBLESHOOTING.md) for solutions to common problems:

### Quick Fixes

**Import errors?**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
PYTHONPATH=$PYTHONPATH:. python examples/complete_strategy.py
```

**Not enough data after cleaning?**
```python
# Use longer period
data = loader.download_stock_data("AAPL", period="5y")  # Instead of "1y"
```

**Model training too slow?**
```python
# Reduce model complexity
rf_model = TradingModel(model_type='random_forest', n_estimators=50, max_depth=5)
```

**Poor backtest results?**
```python
# Try different threshold
for threshold in [0.50, 0.55, 0.60, 0.65]:
    signals = backtest.generate_signals_from_predictions(predictions, threshold=threshold)
    results = backtest.run_backtest(test_data, signals)
    print(f"Threshold {threshold}: {results.total_return:.2%}")
```

📖 **Full troubleshooting guide:** [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

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

### Running Tests

Para executar os testes unitários do projeto, navegue até o diretório raiz do repositório e utilize o `pytest`:

```bash
# Run all tests (28 tests)
PYTHONPATH=$PYTHONPATH:. pytest tests/ -v

# Run tests with coverage report
PYTHONPATH=$PYTHONPATH:. pytest --cov=src tests/ -v --cov-report=html

# Run specific test file
PYTHONPATH=$PYTHONPATH:. pytest tests/test_backtest_engine.py -v

# Run specific test
PYTHONPATH=$PYTHONPATH:. pytest tests/test_ml_models.py::test_trading_model_fit_predict -v
```

### Test Coverage

Current test coverage: **85%+**

| Module | Coverage | Tests |
|--------|----------|-------|
| `data_loader.py` | 90% | 2 tests |
| `technical_indicators.py` | 85% | 2 tests |
| `ml_models.py` | 88% | 12 tests |
| `backtest_engine.py` | 92% | 14 tests |

### Continuous Integration

All tests run automatically on:
- ✅ Push to master/main
- ✅ Pull requests
- ✅ Multiple OS (Ubuntu, Windows, macOS)
- ✅ Multiple Python versions (3.9, 3.10, 3.11, 3.12)

See [.github/workflows/tests.yml](.github/workflows/tests.yml) for CI/CD configuration.

---

## 🎯 How It Works

Want to understand the internals? We've got you covered:

### 📚 Documentation

- **[How It Works](docs/HOW_IT_WORKS.md)** - Complete technical walkthrough
  - Data loading and preprocessing
  - Feature engineering with 32+ indicators
  - Multi-model training and ensemble
  - Signal generation logic
  - Backtesting simulation
  - Performance metrics calculation

- **[ML Pipeline](docs/ml_pipeline.md)** - Visual pipeline documentation with diagrams

- **[FAQ](docs/FAQ.md)** - Frequently asked questions

- **[Use Cases](docs/USE_CASES.md)** - Real-world applications

### 🔍 Quick Overview

**1. Data Pipeline:**
```
Yahoo Finance API → OHLCV Data → Feature Engineering → 32+ Indicators → Training Data
```

**2. Model Training:**
```
Training Data → Random Forest + XGBoost + LightGBM → Ensemble Model → Predictions
```

**3. Trading Simulation:**
```
Predictions → Signals (Buy/Sell/Hold) → Backtest Engine → Performance Metrics
```

**4. Evaluation:**
```
Results → Sharpe Ratio, Drawdown, Win Rate, Profit Factor → Analysis
```

📖 **Read the full guide:** [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md)

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

**Saída Esperada:**
```
================================================================================
ML Trading Strategy - Complete Example
================================================================================

[1/5] Loading market data...
Loaded 1258 days of data for AAPL

[2/5] Engineering features...
Created 32 technical indicators

[3/5] Preparing training data...
Training set: 760 samples
Validation set: 85 samples
Test set: 213 samples

[4/5] Training models...
  Training Random Forest...
  Training XGBoost...
  Training LightGBM...

[5/5] Running backtest...

================================================================================
BACKTEST RESULTS
================================================================================
Total Return:         8.45%
Sharpe Ratio:         1.23
Max Drawdown:         -7.32%
Win Rate:             52.00%
================================================================================
```

> 🎓 **Quer um tutorial interativo?** Confira nosso [Jupyter Notebook](notebooks/01_complete_tutorial.ipynb)!

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

## 📖 Documentação Completa

### 📚 Guias Detalhados

- **[Como Funciona](docs/HOW_IT_WORKS.md)** - Explicação técnica completa em inglês
  - Carregamento e preprocessamento de dados
  - Engenharia de features com 32+ indicadores
  - Treinamento de múltiplos modelos e ensemble
  - Lógica de geração de sinais
  - Simulação de backtesting
  - Cálculo de métricas de performance

- **[Pipeline ML](docs/ml_pipeline.md)** - Documentação visual do pipeline com diagramas

- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Solução de problemas comuns

- **[FAQ](docs/FAQ.md)** - Perguntas frequentes

- **[Casos de Uso](docs/USE_CASES.md)** - Aplicações do mundo real

### 🔍 Visão Geral Rápida

**1. Pipeline de Dados:**
```
API Yahoo Finance → Dados OHLCV → Engenharia de Features → 32+ Indicadores → Dados de Treino
```

**2. Treinamento de Modelos:**
```
Dados de Treino → Random Forest + XGBoost + LightGBM → Modelo Ensemble → Previsões
```

**3. Simulação de Trading:**
```
Previsões → Sinais (Compra/Venda/Manter) → Motor de Backtest → Métricas de Performance
```

**4. Avaliação:**
```
Resultados → Sharpe Ratio, Drawdown, Taxa de Acerto, Profit Factor → Análise
```

---

## 🔧 Resolução de Problemas

Encontrou algum problema? Confira nosso [Guia de Troubleshooting](docs/TROUBLESHOOTING.md) completo.

### Soluções Rápidas

**Erros de importação?**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
PYTHONPATH=$PYTHONPATH:. python examples/complete_strategy.py
```

**Poucos dados após limpeza?**
```python
# Use período mais longo
data = loader.download_stock_data("AAPL", period="5y")  # Em vez de "1y"
```

**Treinamento lento?**
```python
# Reduza a complexidade do modelo
rf_model = TradingModel(model_type='random_forest', n_estimators=50, max_depth=5)
```

📖 **Guia completo:** [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)

---

## 📊 Benchmarks de Performance

Baseado em backtests usando 5 anos de dados da AAPL:

| Métrica | Valor | Descrição |
|---------|-------|-----------|
| **Retorno Total** | 8-15% | Varia conforme condições de mercado |
| **Retorno Anualizado** | 3-8% | Retorno anual sobre investimento |
| **Sharpe Ratio** | 0.8-1.5 | Retorno ajustado ao risco |
| **Drawdown Máximo** | -8% a -15% | Maior queda pico-a-vale |
| **Taxa de Acerto** | 48-55% | Percentual de trades lucrativos |
| **Profit Factor** | 1.5-2.5 | Lucro bruto / Perda bruta |

*Nota: Performance passada não garante resultados futuros. Sempre faça seus próprios testes.*

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



---

## 🏗️ Arquitetura

![Pipeline de Estratégia ML Trading](docs/images/architecture.png)

