"""
Technical Indicators Feature Engineering Module

This module provides comprehensive technical indicator calculations
for financial time series data, optimized for machine learning applications.
"""

from typing import Tuple

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """
    A comprehensive class for calculating technical indicators used in trading strategies.
    """

    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return data.rolling(window=period).mean()

    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)

        Args:
            data: Price series
            period: RSI period (default: 14)

        Returns:
            RSI values (0-100)
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(
        data: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = data.ewm(span=fast_period, adjust=False).mean()
        ema_slow = data.ewm(span=slow_period, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(
        data: pd.Series, period: int = 20, num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands

        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        middle_band = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()

        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)

        return upper_band, middle_band, lower_band

    @staticmethod
    def calculate_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR)

        Measures market volatility
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    @staticmethod
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator

        Returns:
            Tuple of (%K line, %D line)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_line = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_line = k_line.rolling(window=d_period).mean()

        return k_line, d_line

    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV)

        Measures buying and selling pressure
        """
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    @staticmethod
    def calculate_adx(
        high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """
        Calculate Average Directional Index (ADX)

        Measures trend strength (0-100)
        """
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate Directional Movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smooth the values
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)

        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    @staticmethod
    def calculate_vwap(
        high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
    ) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP)
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap

    @classmethod
    def add_all_features(
        cls,
        df: pd.DataFrame,
        price_col: str = "close",
        high_col: str = "high",
        low_col: str = "low",
        volume_col: str = "volume",
    ) -> pd.DataFrame:
        """
        Add all technical indicators to a dataframe

        Args:
            df: DataFrame with OHLCV data
            price_col: Name of close price column
            high_col: Name of high price column
            low_col: Name of low price column
            volume_col: Name of volume column

        Returns:
            DataFrame with all technical indicators added
        """
        result = df.copy()

        # Moving Averages
        for period in [5, 10, 20, 50, 200]:
            result[f"sma_{period}"] = cls.calculate_sma(df[price_col], period)
            result[f"ema_{period}"] = cls.calculate_ema(df[price_col], period)

        # RSI
        result["rsi_14"] = cls.calculate_rsi(df[price_col], 14)

        # MACD
        macd, signal, hist = cls.calculate_macd(df[price_col])
        result["macd"] = macd
        result["macd_signal"] = signal
        result["macd_hist"] = hist

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = cls.calculate_bollinger_bands(df[price_col])
        result["bb_upper"] = bb_upper
        result["bb_middle"] = bb_middle
        result["bb_lower"] = bb_lower
        result["bb_width"] = (bb_upper - bb_lower) / bb_middle

        # ATR
        result["atr_14"] = cls.calculate_atr(
            df[high_col], df[low_col], df[price_col], 14
        )

        # Stochastic
        stoch_k, stoch_d = cls.calculate_stochastic(
            df[high_col], df[low_col], df[price_col]
        )
        result["stoch_k"] = stoch_k
        result["stoch_d"] = stoch_d

        # OBV
        result["obv"] = cls.calculate_obv(df[price_col], df[volume_col])

        # ADX
        result["adx_14"] = cls.calculate_adx(
            df[high_col], df[low_col], df[price_col], 14
        )

        # VWAP
        result["vwap"] = cls.calculate_vwap(
            df[high_col], df[low_col], df[price_col], df[volume_col]
        )

        # Price momentum
        for period in [1, 5, 10, 20]:
            result[f"return_{period}d"] = df[price_col].pct_change(period)

        # Volatility
        for period in [5, 10, 20]:
            result[f"volatility_{period}d"] = (
                df[price_col].pct_change().rolling(period).std()
            )

        return result
