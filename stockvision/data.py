
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INSTRUMENTS_FILE = PROJECT_ROOT / "instruments.csv"

load_dotenv(PROJECT_ROOT / ".env")


# --------------------------------------------------
# KOTAK NEO CLIENT
# --------------------------------------------------

def create_neo_client():
    """
    Create and return a Kotak Neo API client.

    Credentials are read from the local .env file.
    Never hardcode API credentials in this module.
    """

    consumer_key = os.getenv("NEO_CONSUMER_KEY")
    environment = os.getenv("NEO_ENVIRONMENT", "prod")

    if not consumer_key or not consumer_key.strip():
        raise ValueError(
            "Kotak Neo consumer key is missing. "
            "Add NEO_CONSUMER_KEY to your local .env file."
        )

    if consumer_key.strip().startswith("YOUR_"):
        raise ValueError(
            "Replace the placeholder consumer key "
            "with your actual Kotak Neo consumer key."
        )

    try:
        from neo_api_client import NeoAPI

    except ImportError as exc:
        raise ImportError(
            "Could not import the Kotak Neo SDK. "
            "Install dependencies using: "
            "pip install -r requirements.txt"
        ) from exc

    return NeoAPI(
        consumer_key=consumer_key.strip(),
        environment=environment.strip()
    )


# --------------------------------------------------
# INSTRUMENT LOOKUP
# --------------------------------------------------

def get_instrument_symbol(
    symbol,
    instruments_file=INSTRUMENTS_FILE
):
    """
    Find the Kotak Neo instrument identifier.

    Example:
        RELIANCE -> nse_cm|1333

    The token must come from a verified instrument record.
    """

    symbol = str(symbol).strip().upper()

    if not symbol:
        raise ValueError("Stock symbol cannot be empty.")

    instruments_file = Path(instruments_file)

    if not instruments_file.exists():
        raise FileNotFoundError(
            f"Instrument file not found: {instruments_file}"
        )

    instruments = pd.read_csv(
        instruments_file,
        dtype=str
    )

    required_columns = {
        "symbol",
        "exchange_segment",
        "instrument_token"
    }

    missing_columns = (
        required_columns - set(instruments.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing columns in instruments.csv: "
            + ", ".join(sorted(missing_columns))
        )

    instruments["symbol"] = (
        instruments["symbol"]
        .fillna("")
        .str.strip()
        .str.upper()
    )

    matches = instruments[
        instruments["symbol"] == symbol
    ]

    if matches.empty:
        raise ValueError(
            f"Instrument '{symbol}' was not found "
            "in instruments.csv. "
            "Use the instrument lookup feature first."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple instrument entries found for {symbol}. "
            "Keep one verified entry for this symbol."
        )

    row = matches.iloc[0]

    exchange_raw = row["exchange_segment"]
    token_raw = row["instrument_token"]

    # Explicitly reject missing or blank values.
    if (
        pd.isna(exchange_raw)
        or pd.isna(token_raw)
        or not str(exchange_raw).strip()
        or not str(token_raw).strip()
    ):
        raise ValueError(
            f"Exchange segment or instrument token "
            f"is missing for {symbol}."
        )

    exchange_segment = str(exchange_raw).strip()
    instrument_token = str(token_raw).strip()

    if instrument_token.lower() in {"nan", "none", "null"}:
        raise ValueError(
            f"Invalid instrument token for {symbol}."
        )

    return f"{exchange_segment}|{instrument_token}"


# --------------------------------------------------
# DATE CONVERSION
# --------------------------------------------------

def _convert_date(value):
    """
    Convert strings, datetime objects or date objects
    into Python date objects.

    Expected string format: YYYY-MM-DD
    """

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    try:
        return date.fromisoformat(str(value))

    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid date '{value}'. "
            "Use YYYY-MM-DD format."
        ) from exc


# --------------------------------------------------
# HISTORICAL CANDLE PARSER
# --------------------------------------------------

def _parse_candles(response):
    """
    Convert Kotak Neo historical candle response
    into a clean OHLCV DataFrame.

    Expected response structure:

    {
        "status": "success",
        "data": {
            "candles": [
                [
                    "2026-08-20T09:15:00+0530",
                    1200,
                    1210,
                    1195,
                    1205,
                    10000,
                    0
                ]
            ]
        }
    }

    Candle positions:
        0 = timestamp
        1 = open
        2 = high
        3 = low
        4 = close
        5 = volume
        6 = open interest
    """

    columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    if not isinstance(response, dict):
        raise RuntimeError(
            "Unexpected response from Kotak Neo "
            "historical data API."
        )

    status = str(
        response.get("status", "")
    ).strip().lower()

    if status not in {"success", "ok"}:

        fault = response.get("fault", {})

        error_message = (
            fault.get("message", "Unknown API error")
            if isinstance(fault, dict)
            else str(fault)
        )

        raise RuntimeError(
            f"Kotak Neo historical API error: "
            f"{error_message}"
        )

    data = response.get("data", {})

    if not isinstance(data, dict):
        raise RuntimeError(
            "Unexpected historical data format returned by Kotak Neo."
        )

    candles = data.get("candles", [])

    if not candles:
        return pd.DataFrame(columns=columns)

    records = []

    for candle in candles:

        if not isinstance(candle, (list, tuple)):
            continue

        if len(candle) < 6:
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

    # Convert ISO timestamps to Indian market time.
    df["date"] = (
        pd.to_datetime(
            df["date"],
            errors="coerce",
            utc=True
        )
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
    )

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

    # Remove invalid records.
    df = df.dropna(
        subset=columns
    )

    # Remove impossible price records.
    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] >= 0)
    ]

    # Sort and remove duplicate timestamps.
    df = (
        df.sort_values("date")
        .drop_duplicates(
            subset="date",
            keep="last"
        )
        .reset_index(drop=True)
    )

    return df[columns]


# --------------------------------------------------
# FETCH DAILY HISTORICAL DATA
# --------------------------------------------------

def fetch_daily_history(
    symbol,
    start_date,
    end_date,
    client=None,
    instruments_file=INSTRUMENTS_FILE
):
    """
    Fetch daily historical OHLCV data for a stock.

    Kotak Neo allows a maximum of 180 calendar days
    per daily historical-data request.

    Longer date ranges are automatically divided
    into smaller requests.

    Returns:
        pandas DataFrame containing:

        date
        open
        high
        low
        close
        volume
    """

    symbol = str(symbol).strip().upper()

    if not symbol:
        raise ValueError("Stock symbol cannot be empty.")

    start_date = _convert_date(start_date)
    end_date = _convert_date(end_date)

    if start_date > end_date:
        raise ValueError(
            "Start date cannot be later than end date."
        )

    if client is None:
        client = create_neo_client()

    # Retrieve the verified exchange segment and token.
    neo_symbol = get_instrument_symbol(
        symbol=symbol,
        instruments_file=instruments_file
    )

    all_data = []

    current_start = start_date

    while current_start <= end_date:

        # 180 calendar days inclusive.
        current_end = min(
            current_start + timedelta(days=179),
            end_date
        )

        print(
            f"Fetching {symbol}: "
            f"{current_start} to {current_end}"
        )

        try:

            response = client.historical_data(
                neosymbol=neo_symbol,
                interval="D",
                from_date=current_start.isoformat(),
                to_date=current_end.isoformat()
            )

        except Exception as exc:
            raise RuntimeError(
                f"Kotak Neo request failed for {symbol} "
                f"between {current_start} and {current_end}: "
                f"{exc}"
            ) from exc

        chunk = _parse_candles(response)

        if not chunk.empty:
            all_data.append(chunk)

        # Continue with the next date window.
        current_start = current_end + timedelta(days=1)

    if not all_data:

        return pd.DataFrame(
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume"
            ]
        )

    # Combine all historical-data chunks.
    result = pd.concat(
        all_data,
        ignore_index=True
    )

    # Sort, remove duplicates and reset index.
    result = (
        result.sort_values("date")
        .drop_duplicates(
            subset="date",
            keep="last"
        )
        .reset_index(drop=True)
    )

    return result


# --------------------------------------------------
# SAVE HISTORICAL DATA
# --------------------------------------------------

def save_historical_data(
    price_df,
    symbol,
    output_dir=None
):
    """
    Save historical OHLCV data as a CSV file.

    Example:
        data/RELIANCE_daily.csv
    """

    symbol = str(symbol).strip().upper()

    if not symbol:
        raise ValueError("Stock symbol cannot be empty.")

    if price_df.empty:
        raise ValueError(
            "Cannot save an empty historical dataset."
        )

    if output_dir is None:
        output_dir = PROJECT_ROOT / "data"

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir / f"{symbol}_daily.csv"
    )

    price_df.to_csv(
        output_file,
        index=False
    )

    return output_file

