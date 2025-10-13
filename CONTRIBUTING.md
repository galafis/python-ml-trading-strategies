# Contributing to ML Trading Strategies

First off, thank you for considering contributing to this project! 🎉

The following is a set of guidelines for contributing to this project. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Pull Requests](#pull-requests)
- [Style Guidelines](#style-guidelines)
  - [Git Commit Messages](#git-commit-messages)
  - [Python Style Guide](#python-style-guide)
  - [Documentation Style Guide](#documentation-style-guide)
- [Development Setup](#development-setup)
- [Testing Guidelines](#testing-guidelines)

## Code of Conduct

This project and everyone participating in it is governed by a Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

### Our Pledge

We are committed to making participation in this project a harassment-free experience for everyone, regardless of level of experience, gender, gender identity and expression, sexual orientation, disability, personal appearance, body size, race, ethnicity, age, religion, or nationality.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues as you might find that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title**
* **Describe the exact steps to reproduce the problem**
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed and what you expected**
* **Include screenshots if relevant**
* **Include your environment details** (OS, Python version, etc.)

**Bug Report Template:**

```markdown
**Description:**
[Clear description of the bug]

**Steps to Reproduce:**
1. [First step]
2. [Second step]
3. [and so on...]

**Expected Behavior:**
[What you expected to happen]

**Actual Behavior:**
[What actually happened]

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python Version: [e.g., 3.9.7]
- Package Version: [e.g., 0.1.0]

**Additional Context:**
[Any other context about the problem]
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title**
* **Provide a step-by-step description of the suggested enhancement**
* **Provide specific examples to demonstrate the enhancement**
* **Describe the current behavior and expected behavior**
* **Explain why this enhancement would be useful**

### Your First Code Contribution

Unsure where to begin contributing? You can start by looking through `good-first-issue` and `help-wanted` issues:

* **Good first issues** - issues which should only require a few lines of code
* **Help wanted issues** - issues which are a bit more involved

### Pull Requests

1. **Fork the repository** and create your branch from `master`
2. **Make your changes** following the style guidelines
3. **Add tests** for your changes
4. **Ensure the test suite passes**
5. **Update documentation** if necessary
6. **Submit a pull request**

**Pull Request Template:**

```markdown
**Description:**
[Brief description of changes]

**Motivation and Context:**
[Why is this change required? What problem does it solve?]

**Type of Change:**
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update

**How Has This Been Tested?**
[Describe the tests you ran]

**Checklist:**
- [ ] My code follows the code style of this project
- [ ] I have added tests to cover my changes
- [ ] All new and existing tests passed
- [ ] I have updated the documentation accordingly
- [ ] My changes generate no new warnings
```

## Style Guidelines

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

**Examples:**
```
Add support for LightGBM model
Fix backtest engine commission calculation
Update README with new examples
Refactor feature engineering module
```

### Python Style Guide

This project follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide with some exceptions:

* **Line length**: Maximum 100 characters (instead of 79)
* **Formatting**: Use `black` for automatic code formatting
* **Import ordering**: Use `isort` for import organization
* **Type hints**: Add type hints to all function signatures
* **Docstrings**: Use Google-style docstrings

**Example:**

```python
def calculate_returns(
    prices: pd.Series,
    period: int = 1,
    method: str = "simple"
) -> pd.Series:
    """
    Calculate returns for a given price series.
    
    Args:
        prices: Series of prices
        period: Number of periods for return calculation
        method: Type of return ('simple' or 'log')
        
    Returns:
        Series of calculated returns
        
    Raises:
        ValueError: If method is not 'simple' or 'log'
        
    Example:
        >>> prices = pd.Series([100, 105, 103, 108])
        >>> returns = calculate_returns(prices, period=1)
    """
    if method == "simple":
        return prices.pct_change(period)
    elif method == "log":
        return np.log(prices / prices.shift(period))
    else:
        raise ValueError(f"Unknown method: {method}")
```

### Documentation Style Guide

* Use **Markdown** for documentation files
* Include **code examples** where appropriate
* Add **diagrams** for complex concepts
* Keep documentation **up-to-date** with code changes
* Use **clear headings** and **table of contents** for long documents

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/galafis/python-ml-trading-strategies.git
   cd python-ml-trading-strategies
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Install in development mode with dev dependencies
   ```

4. **Install pre-commit hooks (recommended):**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Testing Guidelines

### Running Tests

```bash
# Run all tests
PYTHONPATH=$PYTHONPATH:. pytest tests/ -v

# Run tests with coverage
PYTHONPATH=$PYTHONPATH:. pytest --cov=src tests/ -v

# Run specific test file
PYTHONPATH=$PYTHONPATH:. pytest tests/test_ml_models.py -v

# Run specific test
PYTHONPATH=$PYTHONPATH:. pytest tests/test_ml_models.py::test_trading_model_fit_predict -v
```

### Writing Tests

* **Test coverage**: Aim for at least 80% code coverage
* **Test naming**: Use descriptive names (e.g., `test_calculate_rsi_with_valid_input`)
* **Test organization**: Group related tests in the same file
* **Use fixtures**: Create reusable test data with pytest fixtures
* **Test edge cases**: Include tests for boundary conditions and error cases

**Example Test:**

```python
import pytest
import pandas as pd
from src.features.technical_indicators import TechnicalIndicators

@pytest.fixture
def sample_prices():
    """Create sample price data for testing"""
    return pd.Series([100, 102, 101, 103, 105, 104, 106])

def test_calculate_sma_valid_input(sample_prices):
    """Test SMA calculation with valid input"""
    sma = TechnicalIndicators.calculate_sma(sample_prices, period=3)
    assert isinstance(sma, pd.Series)
    assert len(sma) == len(sample_prices)
    assert not pd.isna(sma.iloc[2])  # First valid value at index 2

def test_calculate_sma_invalid_period(sample_prices):
    """Test SMA calculation with invalid period"""
    with pytest.raises(ValueError):
        TechnicalIndicators.calculate_sma(sample_prices, period=-1)
```

### Code Quality Checks

Before submitting a pull request, run these checks:

```bash
# Format code with black
black src/ tests/ examples/

# Sort imports
isort src/ tests/ examples/

# Run linter
flake8 src/ tests/ examples/

# Type checking (optional but recommended)
mypy src/
```

## Additional Notes

### Issue and Pull Request Labels

* **bug**: Something isn't working
* **enhancement**: New feature or request
* **documentation**: Improvements or additions to documentation
* **good first issue**: Good for newcomers
* **help wanted**: Extra attention is needed
* **question**: Further information is requested

### Getting Help

If you need help, you can:

* Open an issue with the `question` label
* Join our community discussions on GitHub
* Check the documentation in the `docs/` folder

## Recognition

Contributors will be recognized in:

* The project README
* Release notes
* GitHub contributors page

Thank you for your contributions! 🙏

---

**Remember:** The goal is to make this project better, so don't be afraid to ask questions or suggest improvements!
