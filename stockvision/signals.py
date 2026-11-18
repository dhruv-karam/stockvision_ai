import pandas as pd


def generate_technical_signal(feature_df):
    """
    Generate a technical market signal from the latest available row.

    The score ranges from -100 to +100:
        Positive = bullish technical bias
        Negative = bearish technical bias
        Near zero = mixed or neutral indicators

    This is an indicator summary, not investment advice.
    """

    required_columns = [
        "close",
        "sma_20",
        "sma_50",
        "sma_200",
        "rsi_14",
        "macd",
        "macd_signal",
        "bb_position",
    ]

    missing = [
        column
        for column in required_columns
        if column not in feature_df.columns
    ]

    if missing:
        raise ValueError(
            "Missing technical indicators: "
            + ", ".join(missing)
        )

    if feature_df.empty:
        raise ValueError("No price data available.")

    latest = feature_df.iloc[-1]

    def valid_number(column):
        value = pd.to_numeric(
            pd.Series([latest[column]]),
            errors="coerce"
        ).iloc[0]

        return pd.notna(value)

    for column in required_columns:
        if not valid_number(column):
            raise ValueError(
                f"Latest value for {column} is unavailable. "
                "Collect more historical data."
            )

    close = float(latest["close"])
    sma_20 = float(latest["sma_20"])
    sma_50 = float(latest["sma_50"])
    sma_200 = float(latest["sma_200"])
    rsi = float(latest["rsi_14"])
    macd = float(latest["macd"])
    macd_signal = float(latest["macd_signal"])
    bb_position = float(latest["bb_position"])

    if close <= 0:
        raise ValueError("Latest closing price must be positive.")

    signals = []

    def add_signal(name, condition):
        """
        condition:
            True  = bullish
            False = bearish
            None  = neutral
        """

        if condition is True:
            direction = "Bullish"
            score = 1

        elif condition is False:
            direction = "Bearish"
            score = -1

        else:
            direction = "Neutral"
            score = 0

        signals.append({
            "indicator": name,
            "direction": direction,
            "score": score,
        })

    # 1. Price compared with 20-day moving average
    add_signal(
        "Price vs SMA 20",
        close > sma_20 if close != sma_20 else None
    )

    # 2. Price compared with 50-day moving average
    add_signal(
        "Price vs SMA 50",
        close > sma_50 if close != sma_50 else None
    )

    # 3. Price compared with 200-day moving average
    add_signal(
        "Price vs SMA 200",
        close > sma_200 if close != sma_200 else None
    )

    # 4. MACD momentum
    add_signal(
        "MACD",
        macd > macd_signal if macd != macd_signal else None
    )

    # 5. RSI momentum
    if 50 <= rsi <= 70:
        rsi_direction = True
    elif rsi < 40 or rsi > 80:
        rsi_direction = False
    else:
        rsi_direction = None

    add_signal("RSI", rsi_direction)

    # 6. Bollinger Band position
    if bb_position > 0.5:
        bb_direction = True
    elif bb_position < 0.5:
        bb_direction = False
    else:
        bb_direction = None

    add_signal("Bollinger Band Position", bb_direction)

    # Overall score
    total_score = sum(
        item["score"] for item in signals
    )

    maximum_score = len(signals)

    normalized_score = (
        total_score / maximum_score
    ) * 100

    if normalized_score >= 35:
        overall_signal = "Bullish Technical Bias"

    elif normalized_score <= -35:
        overall_signal = "Bearish Technical Bias"

    else:
        overall_signal = "Mixed / Neutral"

    # RSI interpretation
    if rsi >= 70:
        rsi_status = "Potentially overbought"
    elif rsi <= 30:
        rsi_status = "Potentially oversold"
    else:
        rsi_status = "Within the conventional range"

    return {
        "overall_signal": overall_signal,
        "score": round(normalized_score, 2),
        "rsi": round(rsi, 2),
        "rsi_status": rsi_status,
        "latest_close": round(close, 2),
        "indicators": pd.DataFrame(signals),
    }
