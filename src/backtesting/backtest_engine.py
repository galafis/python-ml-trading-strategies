"""
Backtesting Engine for Trading Strategies

Provides comprehensive backtesting functionality with performance metrics.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResults:
    """Container for backtest results"""

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    equity_curve: pd.Series
    trades: pd.DataFrame


class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ):
        """
        Initialize backtest engine

        Args:
            initial_capital: Starting capital
            commission: Commission rate (e.g., 0.001 = 0.1%)
            slippage: Slippage rate (e.g., 0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run_backtest(
        self, data: pd.DataFrame, signals: pd.Series, price_col: str = "close"
    ) -> BacktestResults:
        """
        Run backtest on trading signals

        Args:
            data: DataFrame with price data
            signals: Series with trading signals (1=buy, 0=hold, -1=sell)
            price_col: Name of price column

        Returns:
            BacktestResults object with performance metrics
        """
        # Align data and signals
        df = data.copy()
        df["signal"] = signals
        df = df.dropna()

        # Check if we have data
        if len(df) == 0:
            raise ValueError("No data available after aligning signals")

        # Initialize tracking variables
        position = 0  # Current position (0=no position, 1=long)
        cash = self.initial_capital
        shares = 0
        equity = []
        trades = []

        # Simulate trading
        for idx, row in df.iterrows():
            price = row[price_col]
            signal = row["signal"]

            # Calculate current equity
            current_equity = cash + (shares * price)
            equity.append(current_equity)

            # Execute trades based on signals
            if signal == 1 and position == 0:  # Buy signal
                # Calculate shares to buy
                effective_price = price * (1 + self.slippage)
                shares = (cash * (1 - self.commission)) / effective_price
                cash = 0
                position = 1

                trades.append(
                    {
                        "date": idx,
                        "type": "BUY",
                        "price": effective_price,
                        "shares": shares,
                        "value": shares * effective_price,
                    }
                )

            elif signal == -1 and position == 1:  # Sell signal
                # Sell all shares
                effective_price = price * (1 - self.slippage)
                cash = shares * effective_price * (1 - self.commission)

                trades.append(
                    {
                        "date": idx,
                        "type": "SELL",
                        "price": effective_price,
                        "shares": shares,
                        "value": shares * effective_price,
                    }
                )

                shares = 0
                position = 0

        # Close any open position at the end
        if position == 1:
            final_price = df[price_col].iloc[-1]
            cash = shares * final_price * (1 - self.commission)
            shares = 0

        # Create equity curve
        equity_curve = pd.Series(equity, index=df.index)

        # Handle case where no equity data was generated
        if len(equity_curve) == 0:
            equity_curve = pd.Series([self.initial_capital], index=[df.index[0]])

        # Calculate performance metrics
        results = self._calculate_metrics(equity_curve, trades)

        return results

    def _calculate_metrics(
        self, equity_curve: pd.Series, trades: list
    ) -> BacktestResults:
        """Calculate performance metrics from equity curve"""

        # Total return
        total_return = (equity_curve.iloc[-1] / self.initial_capital) - 1

        # Annualized return (assuming 252 trading days per year)
        n_days = len(equity_curve)
        years = n_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Daily returns
        returns = equity_curve.pct_change().dropna()

        # Sharpe ratio (assuming 252 trading days, 0% risk-free rate)
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Trade statistics
        trades_df = pd.DataFrame(trades)

        if len(trades_df) > 0:
            # Calculate P&L for each trade pair
            buy_trades = trades_df[trades_df["type"] == "BUY"]
            sell_trades = trades_df[trades_df["type"] == "SELL"]

            if len(buy_trades) > 0 and len(sell_trades) > 0:
                n_trades = min(len(buy_trades), len(sell_trades))

                pnl = []
                for i in range(n_trades):
                    buy_value = buy_trades.iloc[i]["value"]
                    sell_value = sell_trades.iloc[i]["value"]
                    pnl.append(sell_value - buy_value)

                # Win rate
                wins = sum(1 for p in pnl if p > 0)
                win_rate = wins / len(pnl) if len(pnl) > 0 else 0

                # Profit factor
                gross_profit = sum(p for p in pnl if p > 0)
                gross_loss = abs(sum(p for p in pnl if p < 0))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            else:
                win_rate = 0
                profit_factor = 0
                n_trades = 0
        else:
            win_rate = 0
            profit_factor = 0
            n_trades = 0

        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=n_trades,
            equity_curve=equity_curve,
            trades=trades_df,
        )

    @staticmethod
    def generate_signals_from_predictions(
        predictions: np.ndarray, threshold: float = 0.5
    ) -> pd.Series:
        """
        Convert model predictions to trading signals

        Args:
            predictions: Model predictions (probabilities or classes)
            threshold: Threshold for buy signal (for probabilities)

        Returns:
            Trading signals (1=buy, 0=hold, -1=sell)
        """
        signals = np.zeros(len(predictions))

        # If predictions are probabilities (2D array from predict_proba)
        if predictions.ndim == 2:
            n_classes = predictions.shape[1]
            if n_classes == 3:
                # 3-class: column 0=down, 1=neutral, 2=up
                buy_proba = predictions[:, 2]
                sell_proba = predictions[:, 0]
            else:
                # Binary: column 0=down, 1=up
                buy_proba = predictions[:, 1]
                sell_proba = predictions[:, 0]
            signals[buy_proba > threshold] = 1
            signals[sell_proba > threshold] = -1
        else:
            # If predictions are class labels (3-class: 0=down, 1=neutral, 2=up)
            signals[predictions == 2] = 1
            signals[predictions == 0] = -1

        return pd.Series(signals)

    @staticmethod
    def plot_results(results: BacktestResults) -> None:
        """
        Plot backtest results

        Args:
            results: BacktestResults object
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Equity curve
        axes[0].plot(results.equity_curve)
        axes[0].set_title("Equity Curve")
        axes[0].set_xlabel("Date")
        axes[0].set_ylabel("Portfolio Value ($)")
        axes[0].grid(True)

        # Drawdown
        returns = results.equity_curve.pct_change()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        axes[1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color="red")
        axes[1].set_title("Drawdown")
        axes[1].set_xlabel("Date")
        axes[1].set_ylabel("Drawdown (%)")
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        print("\n=== Backtest Results ===")
        print(f"Total Return: {results.total_return:.2%}")
        print(f"Annualized Return: {results.annualized_return:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        print(f"Win Rate: {results.win_rate:.2%}")
        print(f"Profit Factor: {results.profit_factor:.2f}")
        print(f"Total Trades: {results.total_trades}")
