from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml-trading-strategies",
    version="0.1.0",
    author="Gabriel Demetrios Lafis",
    author_email="gabriel.lafis@example.com",
    description="Machine Learning Trading Strategies with Advanced Feature Engineering and Backtesting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/galafis/python-ml-trading-strategies",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "yfinance>=0.2.28",
        "matplotlib>=3.7.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
        ],
    },
)
