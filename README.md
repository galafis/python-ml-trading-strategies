# 🤖 Python Ml Trading Strategies

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit-learn-1.4-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](#english) | [Português](#português)

---

## English

### 🎯 Overview

**Python Ml Trading Strategies** — Machine Learning framework for quantitative trading strategies with feature engineering, backtesting, and ensemble models

Total source lines: **2,142** across **17** files in **1** language.

### ✨ Key Features

- **Production-Ready Architecture**: Modular, well-documented, and following best practices
- **Comprehensive Implementation**: Complete solution with all core functionality
- **Clean Code**: Type-safe, well-tested, and maintainable codebase
- **Easy Deployment**: Docker support for quick setup and deployment

### 🚀 Quick Start

#### Prerequisites
- Python 3.12+


#### Installation

1. **Clone the repository**
```bash
git clone https://github.com/galafis/python-ml-trading-strategies.git
cd python-ml-trading-strategies
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```





### 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run with verbose output
pytest -v
```

### 📁 Project Structure

```
python-ml-trading-strategies/
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── images/
│   ├── AUDIT_SUMMARY.md
│   ├── FAQ.md
│   ├── USE_CASES.md
│   └── ml_pipeline.md
├── examples/
│   ├── complete_strategy.py
│   └── simple_strategy_synthetic_data.py
├── notebooks/
│   └── README.md
├── src/
│   ├── backtesting/
│   │   ├── __init__.py
│   │   └── backtest_engine.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── technical_indicators.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── ml_models.py
│   ├── strategies/
│   │   └── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   └── __init__.py
├── tests/
│   ├── test_backtest_engine.py
│   ├── test_data_loader.py
│   ├── test_ml_models.py
│   └── test_technical_indicators.py
├── CHANGELOG.md
├── CONTRIBUTING.md
├── README.md
├── requirements.txt
└── setup.py
```

### 🛠️ Tech Stack

| Technology | Usage |
|------------|-------|
| Python | 17 files |

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 👤 Author

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)

---

## Português

### 🎯 Visão Geral

**Python Ml Trading Strategies** — Machine Learning framework for quantitative trading strategies with feature engineering, backtesting, and ensemble models

Total de linhas de código: **2,142** em **17** arquivos em **1** linguagem.

### ✨ Funcionalidades Principais

- **Arquitetura Pronta para Produção**: Modular, bem documentada e seguindo boas práticas
- **Implementação Completa**: Solução completa com todas as funcionalidades principais
- **Código Limpo**: Type-safe, bem testado e manutenível
- **Fácil Implantação**: Suporte Docker para configuração e implantação rápidas

### 🚀 Início Rápido

#### Pré-requisitos
- Python 3.12+


#### Instalação

1. **Clone the repository**
```bash
git clone https://github.com/galafis/python-ml-trading-strategies.git
cd python-ml-trading-strategies
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```




### 🧪 Testes

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov --cov-report=html

# Run with verbose output
pytest -v
```

### 📁 Estrutura do Projeto

```
python-ml-trading-strategies/
├── data/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── images/
│   ├── AUDIT_SUMMARY.md
│   ├── FAQ.md
│   ├── USE_CASES.md
│   └── ml_pipeline.md
├── examples/
│   ├── complete_strategy.py
│   └── simple_strategy_synthetic_data.py
├── notebooks/
│   └── README.md
├── src/
│   ├── backtesting/
│   │   ├── __init__.py
│   │   └── backtest_engine.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── technical_indicators.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── ml_models.py
│   ├── strategies/
│   │   └── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   └── __init__.py
├── tests/
│   ├── test_backtest_engine.py
│   ├── test_data_loader.py
│   ├── test_ml_models.py
│   └── test_technical_indicators.py
├── CHANGELOG.md
├── CONTRIBUTING.md
├── README.md
├── requirements.txt
└── setup.py
```

### 🛠️ Stack Tecnológica

| Tecnologia | Uso |
|------------|-----|
| Python | 17 files |

### 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

### 👤 Autor

**Gabriel Demetrios Lafis**

- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Demetrios Lafis](https://linkedin.com/in/gabriel-demetrios-lafis)
