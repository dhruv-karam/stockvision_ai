import numpy as np
import pandas as pd


def calculate_risk_metrics(
    price_df,
    risk_free_rate=0.06,
    confidence_level=0.95,
):
    """
    Calculate historical risk and return metrics.

    Args:
        price_df: DataFrame containing date and close columns.
        risk_free_rate: Annual risk-free rate as a decimal.
                        Default is 6%, an illustrative assumption.
        confidence_level: Historical VaR confidence level.

    Returns:
        Dictionary of historical risk metrics.
    """

    required = {"date", "close"}
    missing = required - set(price_df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "Confidence level must be between 0 and 1."
        )

    if risk_free_rate <= -1:
        raise ValueError(
            "Risk-free rate must be greater than -100%."
        )

    df = price_df[["date", "close"]].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce"
    )

    df = (
        df.dropna()
        .sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )

    if len(df) < 3:
        raise ValueError(
            "At least three valid closing-price observations "
            "are required."
        )

    if (df["close"] <= 0).any():
        raise ValueError(
            "Closing prices must be positive."
        )

    returns = df["close"].pct_change().dropna()

    if len(returns) < 2:
        raise ValueError(
            "Not enough valid daily returns to calculate risk."
        )

    # Daily risk-free return derived from the annual assumption.
    daily_risk_free_rate = (
        (1 + risk_free_rate) ** (1 / 252)
    ) - 1

    excess_returns = returns - daily_risk_free_rate

    # Annualized historical return.
    trading_periods = len(returns)

    annualized_return = (
        (df["close"].iloc[-1] / df["close"].iloc[0])
        ** (252 / trading_periods)
    ) - 1

    # Annualized volatility.
    daily_volatility = returns.std(ddof=1)
    annualized_volatility = (
        daily_volatility * np.sqrt(252)
    )

    # Sharpe ratio.
    if daily_volatility > 0:
        sharpe_ratio = (
            excess_returns.mean()
            / daily_volatility
        ) * np.sqrt(252)
    else:
        sharpe_ratio = np.nan

    # Sortino ratio using downside deviation.
    downside_returns = np.minimum(
        excess_returns.to_numpy(),
        0
    )

    downside_deviation = np.sqrt(
        np.mean(downside_returns ** 2)
    )

    if downside_deviation > 0:
        sortino_ratio = (
            excess_returns.mean()
            / downside_deviation
        ) * np.sqrt(252)
    else:
        sortino_ratio = np.nan

    # Maximum drawdown.
    running_peak = df["close"].cummax()

    drawdown = (
        df["close"] / running_peak
    ) - 1

    maximum_drawdown = drawdown.min()

    # Historical Value at Risk and Conditional VaR.
    daily_var = np.quantile(
        returns,
        1 - confidence_level
    )

    tail_returns = returns[
        returns <= daily_var
    ]

    daily_cvar = (
        tail_returns.mean()
        if not tail_returns.empty
        else daily_var
    )

    # Positive values represent estimated loss magnitudes.
    return {
        "observations": int(len(returns)),
        "annualized_return_pct": float(
            annualized_return * 100
        ),
        "annualized_volatility_pct": float(
            annualized_volatility * 100
        ),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "maximum_drawdown_pct": float(
            maximum_drawdown * 100
        ),
        "daily_var_pct": float(
            -daily_var * 100
        ),
        "daily_cvar_pct": float(
            -daily_cvar * 100
        ),
        "confidence_level_pct": confidence_level * 100,
        "risk_free_rate_pct": risk_free_rate * 100,
    }
