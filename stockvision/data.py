
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INSTRUMENTS_FILE = PROJECT_ROOT / "instruments.csv"

DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")


# ==================================================
# CREATE KOTAK NEO CLIENT
# ==================================================

def create_neo_client():
    """
    Create Kotak Neo API client.

    Consumer key is loaded from the local .env file
    or from environment variables supplied by Streamlit Cloud.
    """

    consumer_key = os.getenv("NEO_CONSUMER_KEY")
    environment = os.getenv("NEO_ENVIRONMENT", "prod")

    if not consumer_key or not consumer_key.strip():
        raise ValueError(
            "Kotak Neo consumer key is missing. "
            "Add NEO_CONSUMER_KEY to your local .env file "
            "or Streamlit Cloud Secrets."
        )

    consumer_key = consumer_key.strip()

    if consumer_key.startswith("YOUR_"):
        raise ValueError(
            "Replace the placeholder with your actual Kotak Neo key."
        )

    try:
        from neo_api_client import NeoAPI

    except ImportError as exc:
        raise ImportError(
            "Kotak Neo SDK is not installed. "
            "Install dependencies using requirements.txt."
        ) from exc

    return NeoAPI(
        consumer_key=consumer_key,
        environment=environment.strip()
    )


# ==================================================
# INSTRUMENT LOOKUP
# ==================================================

def get_instrument_symbol(
    symbol,
    instruments_file=INSTRUMENTS_FILE
):
    """
    Read the verified exchange segment and instrument token.

    Example:
        RELIANCE -> nse_cm|1333

    Never guess instrument tokens.
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

    required = {
        "symbol",
        "exchange_segment",
        "instrument_token"
    }

    missing = required - set(instruments.columns)

    if missing:
        raise ValueError(
            "Missing columns in instruments.csv: "
            + ", ".join(sorted(missing))
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
            f"{symbol} was not found in instruments.csv. "
            "Use the NSE instrument lookup feature first."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple instruments found for {symbol}. "
            "Keep one verified NSE equity record."
        )

    row = matches.iloc[0]

    exchange_raw = row["exchange_segment"]
    token_raw = row["instrument_token"]

    if (
        pd.isna(exchange_raw)
        or pd.isna(token_raw)
        or not str(exchange_raw).strip()
        or not str(token_raw).strip()
    ):
        raise ValueError(
            f"Exchange segment or token is missing for {symbol}."
        )

    exchange = str(exchange_raw).strip()
    token = str(token_raw).strip()

    if token.lower() in {"nan", "none", "null"}:
        raise ValueError(
            f"Invalid instrument token for {symbol}."
        )

    if not token.isdigit() or int(token) <= 0:
        raise ValueError(
            f"Invalid instrument token for {symbol}: {token}"
        )

    return f"{exchange}|{token}"


# ==================================================
# DATE CONVERSION
# ==================================================

def _convert_date(value):
    """Convert input to a Python date object."""

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    try:
        return date.fromisoformat(str(value))

    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid date: {value}. Expected YYYY-MM-DD."
        ) from exc


# ==================================================
# API ERROR EXTRACTION
# ==================================================

def _extract_api_error(response):
    """
    Extract useful error messages from Kotak responses.

    Supports:
        status / fault
        stat / error
        errMsg / emsg / desc
    """

    messages = []

    fault = response.get("fault")

    if isinstance(fault, dict):

        message = fault.get("message")
        code = fault.get("code")

        if message:
            messages.append(str(message))

        if code:
            messages.append(f"Error code: {code}")

    elif fault:
        messages.append(str(fault))

    errors = response.get("error")

    if isinstance(errors, list):

        for item in errors:

            if isinstance(item, dict):

                message = item.get("message")
                code = item.get("code")

                if message:
                    messages.append(str(message))

                if code:
                    messages.append(f"Error code: {code}")

            elif item:
                messages.append(str(item))

    elif isinstance(errors, dict):

        message = errors.get("message")
        code = errors.get("code")

        if message:
            messages.append(str(message))

        if code:
            messages.append(f"Error code: {code}")

    elif errors:
        messages.append(str(errors))

    for key in (
        "errMsg",
        "emsg",
        "desc",
        "message",
    ):

        value = response.get(key)

        if value:
            messages.append(str(value))

    code = (
        response.get("stCode")
        or response.get("code")
    )

    if code:
        messages.append(f"Status code: {code}")

    if not messages:
        messages.append(
            "No detailed error message was supplied. "
            f"Response fields: {list(response.keys())}"
        )

    return "; ".join(dict.fromkeys(messages))


# ==================================================
# PARSE HISTORICAL CANDLES
# ==================================================

def _parse_candles(response):
    """
    Parse Kotak Neo historical candle response.

    Supports:
        status = success

    And:
        stat = Ok

    Candle format:
        [
            timestamp,
            open,
            high,
            low,
            close,
            volume,
            open_interest
        ]
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
            "Unexpected Kotak response type: "
            f"{type(response).__name__}"
        )

    status = str(
        response.get("status", "")
    ).strip().lower()

    stat = str(
        response.get("stat", "")
    ).strip().lower()

    successful_statuses = {
        "success",
        "ok",
    }

    failed_statuses = {
        "error",
        "not_ok",
        "failed",
        "failure",
    }

    # Accept both documented success fields.
    is_success = (
        status in successful_statuses
        or stat in successful_statuses
    )

    # Explicit errors must never be treated as success.
    explicit_failure = (
        status in failed_statuses
        or stat in failed_statuses
    )

    if explicit_failure or not is_success:

        message = _extract_api_error(response)

        raise RuntimeError(
            f"Kotak Neo historical API error: {message}"
        )

    data = response.get("data", {})

    if not isinstance(data, dict):
        raise RuntimeError(
            "Kotak Neo returned an unexpected data structure."
        )

    candles = data.get("candles", [])

    if not isinstance(candles, list):
        raise RuntimeError(
            "Historical candles were not returned as a list."
        )

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
            "volume": candle[5],
        })

    if not records:
        raise RuntimeError(
            "Kotak returned candles, but their OHLCV format "
            "was not recognised."
        )

    df = pd.DataFrame(records)

    # Convert timestamps to Indian market time.
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

    df = df.dropna(
        subset=columns
    )

    # Remove invalid OHLCV records.
    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] >= 0)
    ]

    return (
        df.sort_values("date")
        .drop_duplicates(
            subset="date",
            keep="last"
        )
        .reset_index(drop=True)[columns]
    )


# ==================================================
# FETCH DAILY HISTORICAL DATA
# ==================================================

def fetch_daily_history(
    symbol,
    start_date,
    end_date,
    client=None,
    instruments_file=INSTRUMENTS_FILE
):
    """
    Download daily OHLCV data from Kotak Neo.

    Daily interval:
        D

    Maximum request window:
        180 calendar days

    Longer periods are automatically split into
    multiple requests.
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

    neo_symbol = get_instrument_symbol(
        symbol=symbol,
        instruments_file=instruments_file
    )

    all_data = []

    current_start = start_date

    while current_start <= end_date:

        # Maximum 180 calendar days inclusive.
        current_end = min(
            current_start + timedelta(days=179),
            end_date
        )

        print(
            f"Downloading {symbol}: "
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
                f"Kotak request failed for {symbol} "
                f"from {current_start} to {current_end}: {exc}"
            ) from exc

        chunk = _parse_candles(response)

        if not chunk.empty:
            all_data.append(chunk)

        current_start = current_end + timedelta(days=1)

    if not all_data:
        return pd.DataFrame(columns=[
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ])

    result = pd.concat(
        all_data,
        ignore_index=True
    )

    result = (
        result.sort_values("date")
        .drop_duplicates(
            subset="date",
            keep="last"
        )
        .reset_index(drop=True)
    )

    return result


# ==================================================
# SAVE HISTORICAL DATA
# ==================================================

def save_historical_data(
    price_df,
    symbol,
    output_dir=None
):
    """Save downloaded historical data as a CSV file."""

    symbol = str(symbol).strip().upper()

    if not symbol:
        raise ValueError("Stock symbol cannot be empty.")

    if price_df.empty:
        raise ValueError(
            "Cannot save an empty historical dataset."
        )

    if output_dir is None:
        output_dir = DATA_DIR

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

