import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume"
]


def calculate_rsi(series, period=14):
    """Calculate the Relative Strength Index."""

    delta = series.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    relative_strength = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + relative_strength))

    # Handle periods with gains but no losses
    rsi = rsi.mask(avg_loss == 0, 100)

    # Handle periods with neither gains nor losses
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)

    return rsi


def calculate_macd(series):
    """Calculate MACD, signal line, and histogram."""

    ema_12 = series.ewm(span=12, adjust=False).mean()
    ema_26 = series.ewm(span=26, adjust=False).mean()

    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal

    return macd, signal, histogram


def calculate_atr(df, period=14):
    """Calculate Average True Range."""

    previous_close = df["close"].shift(1)

    high_low = df["high"] - df["low"]
    high_previous_close = (df["high"] - previous_close).abs()
    low_previous_close = (df["low"] - previous_close).abs()

    true_range = pd.concat(
        [
            high_low,
            high_previous_close,
            low_previous_close
        ],
        axis=1
    ).max(axis=1)

    return true_range.rolling(period).mean()


def build_features(price_df):
    """
    Generate technical indicators and model features.

    Input:
        DataFrame containing date, open, high, low, close, volume.

    Output:
        DataFrame containing original data and calculated features.

    Rows with incomplete indicator history are retained. The modeling
    module will remove rows that cannot be used for training.
    """

    missing = set(REQUIRED_COLUMNS) - set(price_df.columns)

    if missing:
        raise ValueError(
            "Missing required price columns: "
            + ", ".join(sorted(missing))
        )

    if price_df.empty:
        raise ValueError("Price data is empty.")

    df = price_df[REQUIRED_COLUMNS].copy()

    # Clean and order the data
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = (
        df.dropna(subset=REQUIRED_COLUMNS)
        .sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )

    if len(df) < 30:
        raise ValueError(
            "At least 30 rows of valid price data are required "
            "to calculate the technical indicators."
        )

    if (df["close"] <= 0).any():
        raise ValueError("Closing prices must be positive.")

    # Daily returns
    df["daily_return"] = df["close"].pct_change()

    df["log_return"] = np.log(
        df["close"] / df["close"].shift(1)
    )

    # Momentum
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)
    df["return_21d"] = df["close"].pct_change(21)

    # Moving averages
    for period in [10, 20, 50, 100, 200]:
        df[f"sma_{period}"] = (
            df["close"].rolling(period).mean()
        )

    # Exponential moving averages
    for period in [12, 26, 50, 100, 200]:
        df[f"ema_{period}"] = (
            df["close"].ewm(
                span=period,
                adjust=False
            ).mean()
        )

    # Relative Strength Index
    df["rsi_14"] = calculate_rsi(
        df["close"],
        period=14
    )

    # MACD indicators
    (
        df["macd"],
        df["macd_signal"],
        df["macd_histogram"]
    ) = calculate_macd(df["close"])

    # Bollinger Bands
    rolling_mean = df["close"].rolling(20).mean()
    rolling_std = df["close"].rolling(20).std()

    df["bb_middle"] = rolling_mean
    df["bb_upper"] = rolling_mean + (2 * rolling_std)
    df["bb_lower"] = rolling_mean - (2 * rolling_std)

    df["bb_width"] = (
        (df["bb_upper"] - df["bb_lower"])
        / df["bb_middle"].replace(0, np.nan)
    )

    df["bb_position"] = (
        (df["close"] - df["bb_lower"])
        / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    )

    # Average True Range
    df["atr_14"] = calculate_atr(df, period=14)

    df["atr_percent"] = (
        df["atr_14"] / df["close"]
    )

    # Annualized historical volatility
    df["volatility_20d"] = (
        df["log_return"]
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    df["volatility_60d"] = (
        df["log_return"]
        .rolling(60)
        .std()
        * np.sqrt(252)
    )

    # Volume indicators
    df["volume_sma_20"] = (
        df["volume"].rolling(20).mean()
    )

    df["volume_ratio"] = (
        df["volume"]
        / df["volume_sma_20"].replace(0, np.nan)
    )

    # On-Balance Volume
    price_direction = np.sign(df["close"].diff()).fillna(0)

    df["obv"] = (
        price_direction * df["volume"]
    ).cumsum()

    # Relative price to moving averages
    for period in [20, 50, 100, 200]:
        column = f"sma_{period}"

        df[f"price_vs_sma_{period}"] = (
            df["close"] / df[column].replace(0, np.nan)
        ) - 1

    return df
