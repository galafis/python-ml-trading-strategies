# Contributing to ML Trading Strategies

🎉 Thank you for your interest in contributing to this project! 🎉

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- pip or conda

### Setting Up Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/python-ml-trading-strategies.git
   cd python-ml-trading-strategies
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Install dev dependencies
   ```

5. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. **Make your changes** in your feature branch
2. **Write or update tests** to cover your changes
3. **Run tests** to ensure everything works
4. **Format your code** using black and isort
5. **Lint your code** using flake8
6. **Commit your changes** with clear, descriptive messages
7. **Push to your fork** and create a Pull Request

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: Maximum 100 characters (not 79)
- **Indentation**: 4 spaces (no tabs)
- **Naming conventions**:
  - Classes: `PascalCase`
  - Functions/methods: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private members: `_leading_underscore`

### Code Formatting

Use `black` for automatic code formatting:

```bash
black src/ tests/ examples/
```

Use `isort` to organize imports:

```bash
isort src/ tests/ examples/
```

### Linting

Run `flake8` to check for code quality issues:

```bash
flake8 src/ tests/ examples/ --max-line-length=100
```

### Type Hints

- Add type hints to all public functions and methods
- Use `typing` module for complex types
- Example:
  ```python
  def calculate_returns(prices: pd.Series, period: int = 1) -> pd.Series:
      """Calculate percentage returns over specified period."""
      return prices.pct_change(period)
  ```

### Documentation

- Write clear, concise docstrings for all public classes and methods
- Use Google-style docstrings:
  ```python
  def example_function(param1: int, param2: str) -> bool:
      """
      Short description of the function.
      
      Longer description if needed, explaining what the function does,
      any important notes, and usage examples.
      
      Args:
          param1: Description of param1
          param2: Description of param2
          
      Returns:
          Description of return value
          
      Raises:
          ValueError: When param1 is negative
      """
  ```

## Testing Guidelines

### Writing Tests

- Write tests for all new features and bug fixes
- Aim for high code coverage (>80%)
- Use descriptive test names: `test_<function>_<scenario>_<expected_result>`
- Use pytest fixtures for common setup

### Running Tests

Run all tests:
```bash
PYTHONPATH=$PYTHONPATH:. pytest tests/ -v
```

Run tests with coverage:
```bash
PYTHONPATH=$PYTHONPATH:. pytest --cov=src tests/ -v
```

Run specific test file:
```bash
PYTHONPATH=$PYTHONPATH:. pytest tests/test_ml_models.py -v
```

Run specific test:
```bash
PYTHONPATH=$PYTHONPATH:. pytest tests/test_ml_models.py::test_trading_model_fit_predict -v
```

### Test Structure

```python
import pytest
from src.module import Class

def test_feature_success():
    """Test successful execution of feature."""
    # Arrange
    instance = Class()
    
    # Act
    result = instance.method()
    
    # Assert
    assert result == expected_value

def test_feature_raises_error():
    """Test that feature raises appropriate error."""
    instance = Class()
    
    with pytest.raises(ValueError, match="error message"):
        instance.method(invalid_input)
```

## Submitting Changes

### Pull Request Process

1. **Ensure all tests pass**
2. **Update documentation** if needed
3. **Add entries to CHANGELOG.md** describing your changes
4. **Push your changes** to your fork
5. **Create a Pull Request** with:
   - Clear title describing the change
   - Detailed description of what changed and why
   - Reference to any related issues
   - Screenshots (if UI changes)

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new features
- [ ] Code coverage maintained or improved

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Code is commented where necessary
- [ ] Documentation updated
- [ ] No new warnings generated
```

## Reporting Bugs

### Before Submitting a Bug Report

- Check existing issues to avoid duplicates
- Verify the bug in the latest version
- Collect relevant information

### Bug Report Template

Use this template when creating a bug report:

```markdown
**Describe the bug**
Clear and concise description

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen

**Screenshots**
If applicable

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.9.7]
- Package version: [e.g., 0.1.0]

**Additional context**
Any other relevant information
```

## Feature Requests

We welcome feature requests! Please:

1. **Check existing issues** to avoid duplicates
2. **Clearly describe the feature** and its use case
3. **Explain why** this feature would be useful
4. **Provide examples** if possible

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
Clear description of the problem

**Describe the solution you'd like**
Clear description of what you want to happen

**Describe alternatives you've considered**
Other solutions or features you've considered

**Additional context**
Screenshots, mockups, or examples
```

## Development Tips

### Best Practices

1. **Keep changes focused**: One feature or bug fix per pull request
2. **Write tests first**: Consider test-driven development (TDD)
3. **Document as you go**: Update docs with code changes
4. **Ask questions**: Open issues for clarification before starting work
5. **Be patient**: Reviews may take time

### Common Pitfalls to Avoid

- Don't commit large binary files
- Don't include credentials or API keys
- Don't break backward compatibility without discussion
- Don't submit incomplete features

### Getting Help

- 💬 **Discussions**: Use GitHub Discussions for questions
- 🐛 **Issues**: Report bugs via GitHub Issues
- 📧 **Email**: Contact maintainers for sensitive issues

## Recognition

Contributors will be:
- Listed in the project's contributors page
- Acknowledged in release notes
- Appreciated by the community! ⭐

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing!** 🙏

Your efforts help make this project better for everyone.
