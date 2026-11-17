import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTRUMENTS_FILE = PROJECT_ROOT / "instruments.csv"

# Load environment variables from the project's .env file
load_dotenv(PROJECT_ROOT / ".env")


def create_neo_client():
    """
    Create and return a Kotak Neo API client.

    The consumer key must be stored locally in the .env file.
    """

    consumer_key = os.getenv("NEO_CONSUMER_KEY")
    environment = os.getenv("NEO_ENVIRONMENT", "prod")

    if not consumer_key or consumer_key.startswith("YOUR_"):
        raise ValueError(
            "Kotak Neo consumer key is missing. "
            "Create a .env file in the project folder and add "
            "your NEO_CONSUMER_KEY."
        )

    try:
        from neo_api_client import NeoAPI
    except ImportError as exc:
        raise ImportError(
            "Could not import the Kotak Neo SDK. "
            "Run: pip install -r requirements.txt"
        ) from exc

    return NeoAPI(
        consumer_key=consumer_key,
        environment=environment
    )


def get_instrument_symbol(symbol, instruments_file=INSTRUMENTS_FILE):
    """
    Find the Kotak Neo instrument identifier for a stock.

    Example return value:
        nse_cm|1333
    """

    symbol = str(symbol).strip().upper()
    instruments_file = Path(instruments_file)

    if not instruments_file.exists():
        raise FileNotFoundError(
            f"Instrument file not found: {instruments_file}"
        )

    instruments = pd.read_csv(instruments_file, dtype=str)

    required_columns = {
        "symbol",
        "exchange_segment",
        "instrument_token"
    }

    missing_columns = required_columns - set(instruments.columns)

    if missing_columns:
        raise ValueError(
            "Missing columns in instruments.csv: "
            + ", ".join(sorted(missing_columns))
        )

    matches = instruments[
        instruments["symbol"].str.strip().str.upper() == symbol
    ]

    if matches.empty:
        raise ValueError(
            f"Instrument '{symbol}' was not found in instruments.csv. "
            "Add its verified exchange segment and instrument token."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple entries found for '{symbol}'. "
            "Keep one instrument entry for this symbol for now."
        )

    row = matches.iloc[0]

    exchange_segment = str(row["exchange_segment"]).strip()
    instrument_token = str(row["instrument_token"]).strip()

    if not exchange_segment or not instrument_token:
        raise ValueError(
            f"Exchange segment or instrument token is missing for {symbol}."
        )

    return f"{exchange_segment}|{instrument_token}"


def _convert_date(value):
    """Convert a string or date object into a Python date."""

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(
            f"Invalid date '{value}'. Use YYYY-MM-DD format."
        ) from exc


def _parse_candles(response):
    """
    Convert the Kotak Neo historical API response into an OHLCV DataFrame.
    """

    if not isinstance(response, dict):
        raise RuntimeError(
            "Unexpected response from Kotak Neo historical data API."
        )

    status = str(response.get("status", "")).lower()

    if status not in {"success", "ok"}:
        raise RuntimeError(
            f"Kotak Neo returned an unsuccessful response: {response}"
        )

    data = response.get("data", {})
    candles = data.get("candles", []) if isinstance(data, dict) else []

    columns = ["date", "open", "high", "low", "close", "volume"]

    if not candles:
        return pd.DataFrame(columns=columns)

    records = []

    for candle in candles:
        if not isinstance(candle, (list, tuple)) or len(candle) < 6:
            continue

        records.append({
            "date": candle[0],
            "open": candle[1],
            "high": candle[2],
            "low": candle[3],
            "close": candle[4],
            "volume": candle[5]
        })

    if not records:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(records)

    # Convert timestamps to Indian market time, without timezone information
    df["date"] = (
        pd.to_datetime(df["date"], errors="coerce", utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
    )

    numeric_columns = ["open", "high", "low", "close", "volume"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=columns)

    df = (
        df.sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )

    return df[columns]


def fetch_daily_history(
    symbol,
    start_date,
    end_date,
    client=None,
    instruments_file=INSTRUMENTS_FILE
):
    """
    Fetch daily historical OHLCV data for a stock.

    Kotak Neo limits daily historical requests to 180 calendar days.
    This function automatically divides longer periods into smaller
    requests and combines the results.

    Returns:
        pandas DataFrame with:
        date, open, high, low, close, volume
    """

    symbol = str(symbol).strip().upper()

    start_date = _convert_date(start_date)
    end_date = _convert_date(end_date)

    if start_date > end_date:
        raise ValueError("start_date cannot be later than end_date.")

    if client is None:
        client = create_neo_client()

    neo_symbol = get_instrument_symbol(
        symbol,
        instruments_file
    )

    all_data = []
    current_start = start_date

    while current_start <= end_date:

        # Maximum request window: 180 calendar days
        current_end = min(
            current_start + timedelta(days=179),
            end_date
        )

        response = client.historical_data(
            neosymbol=neo_symbol,
            interval="D",
            from_date=current_start.isoformat(),
            to_date=current_end.isoformat()
        )

        chunk = _parse_candles(response)

        if not chunk.empty:
            all_data.append(chunk)

        # Move to the day after the previous request's end date
        current_start = current_end + timedelta(days=1)

    if not all_data:
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume"]
        )

    result = pd.concat(all_data, ignore_index=True)

    result = (
        result.sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )

    return result
