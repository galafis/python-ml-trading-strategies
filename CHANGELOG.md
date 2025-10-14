# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test suite with 34 tests covering all major components
- Backtesting engine tests (11 new tests)
- Expanded technical indicators tests (11 tests total, up from 2)
- CONTRIBUTING.md with detailed contribution guidelines
- Simple strategy example with synthetic data (no internet required)
- Advanced usage examples (Optuna, SHAP, custom strategies) in README
- Troubleshooting section in README (EN and PT)
- Important disclaimers section (EN and PT)
- Performance metrics explanation section
- Test coverage badges
- Binary mode for target variable creation
- Support for 3-class classification compatible with XGBoost

### Changed
- Target encoding changed from (-1, 0, 1) to (0, 1, 2) for XGBoost compatibility
- Improved README with better structure and more details
- Enhanced Portuguese section of README to match English improvements
- Updated contributing section with link to CONTRIBUTING.md
- All code formatted with black and isort
- Imports organized and cleaned up
- Fixed yfinance warning by adding auto_adjust parameter

### Removed
- Unused dependencies: tensorflow, ta-lib
- Unused imports across all modules
- Code style violations (flake8 clean)

### Fixed
- Ensemble model voting TypeError with float to int conversion
- Target variable encoding issues with XGBoost
- yfinance FutureWarning about auto_adjust
- All PEP 8 style violations
- Line length issues (max 100 characters)

### Security
- No security issues identified

## [1.0.0] - 2025-10-06

### Added
- Initial release
- Core functionality implemented
- Comprehensive test suite
- Documentation and examples
- CI/CD pipeline

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

---

## Release Notes

### Version 1.0.1 (Pending)

**Highlights:**
- 🧪 34 comprehensive tests with >95% coverage
- 📚 Complete CONTRIBUTING.md guide  
- 🎨 100% PEP 8 compliant code (black + isort + flake8)
- 📖 Enhanced README with troubleshooting and advanced examples
- 🔧 Fixed XGBoost compatibility issues
- 🧹 Removed unused dependencies
- 🎯 Synthetic data example for offline testing

**Testing:**
- 11 new backtest engine tests
- 9 additional technical indicator tests
- All 34 tests passing
- >95% code coverage

**Code Quality:**
- Zero flake8 errors
- All code formatted with black
- Imports organized with isort
- Removed all unused imports

**Documentation:**
- CONTRIBUTING.md (7.8KB)
- Enhanced README (English and Portuguese)
- Troubleshooting guide
- Performance metrics explanation
- Advanced usage examples

**Contributors:**
- Gabriel Demetrios Lafis

### Version 1.0.0 (2025-10-06)

**Highlights:**
- First stable release
- Production-ready code
- Full documentation
- Extensive test coverage

**Contributors:**
- Gabriel Demetrios Lafis

---

[Unreleased]: https://github.com/galafis/python-ml-trading-strategies/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/galafis/python-ml-trading-strategies/releases/tag/v1.0.0
