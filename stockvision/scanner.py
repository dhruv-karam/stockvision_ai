from datetime import date, timedelta

import numpy as np
import pandas as pd

from stockvision.data import (
    create_neo_client,
    fetch_daily_history,
)

from stockvision.features import build_features

from stockvision.signals import generate_technical_signal

from stockvision.risk import calculate_risk_metrics


def scan_watchlist(
    watchlist_df,
    lookback_days=500,
    client=None,
):
    """
    Scan all stocks in a watchlist.

    Calculates:
    - Latest closing price
    - Daily price change
    - RSI
    - Technical score and bias
    - Annualized historical volatility
    - Maximum historical drawdown

    Returns:
        DataFrame containing one result per stock.

    Each stock is processed independently. A failure for one
    stock will not prevent the remaining stocks from scanning.
    """

    if "symbol" not in watchlist_df.columns:
        raise ValueError(
            "Watchlist must contain a symbol column."
        )

    if watchlist_df.empty:
        raise ValueError("Your watchlist is empty.")

    if lookback_days < 250:
        raise ValueError(
            "Use at least 250 calendar days of historical data "
            "for the technical indicators."
        )

    if client is None:
        client = create_neo_client()

    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    results = []

    symbols = (
        watchlist_df["symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .drop_duplicates()
        .tolist()
    )

    for symbol in symbols:

        try:
            # Fetch historical prices.
            price_df = fetch_daily_history(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                client=client,
            )

            if price_df.empty or len(price_df) < 220:
                raise ValueError(
                    "Insufficient historical data for a "
                    "200-day moving average."
                )

            # Calculate technical indicators.
            feature_df = build_features(price_df)

            latest = feature_df.iloc[-1]

            previous_close = float(
                feature_df["close"].iloc[-2]
            )

            latest_close = float(latest["close"])

            daily_change_pct = (
                (latest_close / previous_close) - 1
            ) * 100

            # Generate technical signal.
            signal = generate_technical_signal(
                feature_df
            )

            # Calculate historical risk.
            risk = calculate_risk_metrics(
                price_df
            )

            results.append({
                "symbol": symbol,
                "status": "Success",
                "latest_price": latest_close,
                "daily_change_pct": daily_change_pct,
                "rsi": float(latest["rsi_14"]),
                "technical_score": signal["score"],
                "technical_bias": signal["overall_signal"],
                "annualized_volatility_pct": (
                    risk["annualized_volatility_pct"]
                ),
                "maximum_drawdown_pct": (
                    risk["maximum_drawdown_pct"]
                ),
                "data_date": str(
                    pd.to_datetime(latest["date"]).date()
                ),
                "error": "",
            })

        except Exception as error:

            results.append({
                "symbol": symbol,
                "status": "Failed",
                "latest_price": np.nan,
                "daily_change_pct": np.nan,
                "rsi": np.nan,
                "technical_score": np.nan,
                "technical_bias": "Unavailable",
                "annualized_volatility_pct": np.nan,
                "maximum_drawdown_pct": np.nan,
                "data_date": "",
                "error": str(error),
            })

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values(
            "technical_score",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    return results_df
