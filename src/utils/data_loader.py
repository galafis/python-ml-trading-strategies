"""
Data Loading and Preprocessing Module

Handles downloading financial data from various sources and preprocessing.
"""

from typing import List, Optional

import pandas as pd
import yfinance as yf


class DataLoader:
    """
    Utility class for loading and preprocessing financial market data.
    """

    @staticmethod
    def download_stock_data(
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y",
    ) -> pd.DataFrame:
        """
        Download stock data from Yahoo Finance

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Period to download if dates not specified (e.g., '1y', '2y', '5y')

        Returns:
            DataFrame with OHLCV data
        """
        if start_date and end_date:
            data = yf.download(
                ticker, start=start_date, end=end_date, progress=False, auto_adjust=True
            )
        else:
            data = yf.download(ticker, period=period, progress=False, auto_adjust=True)

        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Rename columns to lowercase
        data.columns = [col.lower() for col in data.columns]

        # Reset index to have date as column
        data = data.reset_index()
        data.columns = [col.lower() for col in data.columns]

        return data

    @staticmethod
    def download_multiple_stocks(
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "2y",
    ) -> dict:
        """
        Download data for multiple stocks

        Returns:
            Dictionary with ticker as key and DataFrame as value
        """
        data_dict = {}

        for ticker in tickers:
            try:
                data = DataLoader.download_stock_data(
                    ticker, start_date, end_date, period
                )
                data_dict[ticker] = data
                print(f"Downloaded {ticker}: {len(data)} rows")
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")

        return data_dict

    @staticmethod
    def prepare_training_data(
        df: pd.DataFrame,
        target_col: str = "target",
        test_size: float = 0.2,
        validation_size: float = 0.1,
    ) -> tuple:
        """
        Split data into train, validation, and test sets

        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Remove NaN values
        df = df.dropna()

        # Split features and target
        # Exclude date columns and target
        exclude_cols = [target_col, "date", "index"]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols]
        y = df[target_col]

        # Calculate split indices
        n = len(df)
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - validation_size))

        # Split data (time series, so no shuffling)
        X_train = X.iloc[:val_idx]
        X_val = X.iloc[val_idx:test_idx]
        X_test = X.iloc[test_idx:]

        y_train = y.iloc[:val_idx]
        y_val = y.iloc[val_idx:test_idx]
        y_test = y.iloc[test_idx:]

        return X_train, X_val, X_test, y_train, y_val, y_test

    @staticmethod
    def create_target_variable(
        df: pd.DataFrame,
        price_col: str = "close",
        horizon: int = 1,
        threshold: float = 0.0,
        binary: bool = False,
    ) -> pd.Series:
        """
        Create target variable for classification

        Args:
            df: DataFrame with price data
            price_col: Name of price column
            horizon: Number of periods to look ahead
            threshold: Minimum return to classify as positive
            binary: If True, create binary classification (0=down/neutral, 1=up)
                   If False, create 3-class (0=down, 1=neutral, 2=up)

        Returns:
            Target series with class labels
        """
        future_return = df[price_col].pct_change(horizon).shift(-horizon)

        if binary:
            # Binary classification: 0=down/neutral, 1=up
            target = pd.Series(index=df.index, dtype=float, name="target")
            target[future_return > threshold] = 1.0
            target[future_return <= threshold] = 0.0
        else:
            # 3-class classification: 0=down, 1=neutral, 2=up (XGBoost compatible)
            target = pd.Series(index=df.index, dtype=float, name="target")
            target[future_return > threshold] = 2.0
            target[future_return < -threshold] = 0.0
            target[
                (future_return >= -threshold) & (future_return <= threshold)
            ] = 1.0

        return target
        return target

    @staticmethod
    def create_regression_target(
        df: pd.DataFrame, price_col: str = "close", horizon: int = 1
    ) -> pd.Series:
        """
        Create regression target variable (future returns)

        Args:
            df: DataFrame with price data
            price_col: Name of price column
            horizon: Number of periods to look ahead

        Returns:
            Future returns series
        """
        future_return = df[price_col].pct_change(horizon).shift(-horizon)
        return future_return
