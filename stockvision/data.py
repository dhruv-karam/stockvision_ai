
import os
import json
import time
import threading

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

# Conservative spacing between historical API requests.
# Override through environment variable if required.
HISTORICAL_REQUEST_GAP = max(
    1.0,
    float(os.getenv("KOTAK_HISTORICAL_REQUEST_GAP", "5"))
)

_last_historical_request = None
_request_lock = threading.Lock()


# ==================================================
# CUSTOM RATE-LIMIT ERROR
# ==================================================

class KotakRateLimitError(RuntimeError):
    """Raised when Kotak Neo rejects a request due to rate limiting."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ==================================================
# CREATE KOTAK NEO CLIENT
# ==================================================

def create_neo_client():
    """
    Create Kotak Neo API client.

    Reads the consumer key from:
    - Local .env file
    - Environment variables
    - Streamlit Cloud Secrets copied into environment variables
    """

    consumer_key = os.getenv("NEO_CONSUMER_KEY")
    environment = os.getenv("NEO_ENVIRONMENT", "prod")

    if not consumer_key or not consumer_key.strip():
        raise ValueError(
            "Kotak Neo consumer key is missing. "
            "Configure NEO_CONSUMER_KEY in your local .env "
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
            "Install dependencies from requirements.txt."
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
    Retrieve the verified Kotak Neo exchange and token.

    Example:
        RELIANCE -> nse_cm|2885

    The token is read from instruments.csv.
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
            "Use the instrument lookup feature first."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple instrument records found for {symbol}."
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

    if not token.isdigit() or int(token) <= 0:
        raise ValueError(
            f"Invalid instrument token for {symbol}: {token}"
        )

    return f"{exchange}|{token}"


# ==================================================
# DATE CONVERSION
# ==================================================

def _convert_date(value):
    """Convert input to Python date."""

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
# NORMALIZE RESPONSE OBJECT
# ==================================================

def _to_dict(value):
    """
    Convert a response body or exception body into a dictionary
    when possible.
    """

    if isinstance(value, dict):
        return value

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    if isinstance(value, str):
        try:
            parsed = json.loads(value)

            if isinstance(parsed, dict):
                return parsed

        except (ValueError, TypeError):
            pass

    return None


def _response_dicts(source):
    """Collect available dictionaries from responses and exceptions."""

    if source is None:
        return []

    result = []

    if isinstance(source, dict):
        result.append(source)

    for attribute in (
        "body",
        "response_body",
        "detail",
    ):
        value = getattr(source, attribute, None)
        parsed = _to_dict(value)

        if parsed is not None:
            result.append(parsed)

    response = getattr(source, "response", None)

    if response is not None:

        for attribute in ("body", "text"):
            value = getattr(response, attribute, None)

            parsed = _to_dict(value)

            if parsed is not None:
                result.append(parsed)

        try:
            parsed = _to_dict(response.json())

            if parsed is not None:
                result.append(parsed)

        except Exception:
            pass

    parsed = _to_dict(source)

    if parsed is not None:
        result.append(parsed)

    return result


# ==================================================
# RATE LIMIT DETECTION
# ==================================================

def _is_rate_limit_error(source):
    """Detect Kotak HTTP 429 responses and rate-limit messages."""

    if source is None:
        return False

    # Check exception attributes.
    for attribute in (
        "status",
        "status_code",
        "code",
    ):
        value = getattr(source, attribute, None)

        try:
            if int(value) == 429:
                return True
        except (TypeError, ValueError):
            pass

    # Check response dictionaries.
    for item in _response_dicts(source):

        for key in (
            "code",
            "status_code",
            "stCode",
        ):
            try:
                if int(item.get(key)) == 429:
                    return True
            except (TypeError, ValueError):
                pass

    message = str(source).lower()

    rate_limit_phrases = [
        "rate limit exceeded",
        "too many requests",
        "too many request received",
        "status_code': 429",
        '"status_code": 429',
    ]

    return any(
        phrase in message
        for phrase in rate_limit_phrases
    )


# ==================================================
# RETRY-AFTER EXTRACTION
# ==================================================

def _get_retry_after_seconds(source):
    """
    Read Retry-After from Kotak rateLimit metadata or HTTP headers.

    Returns None when the API does not provide a usable value.
    """

    for item in _response_dicts(source):

        # Kotak SDK exposes these headers under rateLimit.
        rate_limit = item.get("rateLimit")

        if isinstance(rate_limit, dict):

            for key, value in rate_limit.items():

                if str(key).lower() in {
                    "retry-after",
                    "x-ratelimit-reset",
                }:

                    try:
                        seconds = float(value)

                        if seconds > 0:
                            return int(seconds)

                    except (TypeError, ValueError):
                        continue

    # Check exception or response HTTP headers.
    possible_headers = []

    headers = getattr(source, "headers", None)

    if isinstance(headers, dict):
        possible_headers.append(headers)

    response = getattr(source, "response", None)

    if response is not None:
        response_headers = getattr(
            response,
            "headers",
            None
        )

        if response_headers is not None:
            possible_headers.append(response_headers)

    for header_set in possible_headers:

        for key, value in header_set.items():

            if str(key).lower() == "retry-after":

                try:
                    seconds = float(value)

                    if seconds > 0:
                        return int(seconds)

                except (TypeError, ValueError):
                    continue

    return None


# ==================================================
# API ERROR MESSAGE
# ==================================================

def _extract_api_error(source):
    """Extract a useful error message from Kotak API responses."""

    messages = []

    for item in _response_dicts(source):

        error = item.get("error")

        if isinstance(error, list):

            for entry in error:

                if isinstance(entry, dict):

                    if entry.get("message"):
                        messages.append(str(entry["message"]))

                    if entry.get("code"):
                        messages.append(
                            f"Error code: {entry['code']}"
                        )

                elif entry:
                    messages.append(str(entry))

        elif isinstance(error, dict):

            if error.get("message"):
                messages.append(str(error["message"]))

            if error.get("code"):
                messages.append(
                    f"Error code: {error['code']}"
                )

        elif error:
            messages.append(str(error))

        fault = item.get("fault")

        if isinstance(fault, dict):

            if fault.get("message"):
                messages.append(str(fault["message"]))

            if fault.get("code"):
                messages.append(
                    f"Error code: {fault['code']}"
                )

        for key in (
            "message",
            "errMsg",
            "emsg",
            "desc",
        ):
            if item.get(key):
                messages.append(str(item[key]))

    if messages:
        return "; ".join(dict.fromkeys(messages))

    return str(source)


# ==================================================
# REQUEST PACING
# ==================================================

def _wait_for_request_slot():
    """
    Ensure historical requests are separated by the configured gap.

    This is process-local pacing, not a guarantee against
    account-wide or provider-side rate limits.
    """

    global _last_historical_request

    with _request_lock:

        if _last_historical_request is not None:

            elapsed = (
                time.monotonic()
                - _last_historical_request
            )

            remaining = HISTORICAL_REQUEST_GAP - elapsed

            if remaining > 0:
                print(
                    f"Waiting {remaining:.1f} seconds "
                    "before next Kotak historical request."
                )

                time.sleep(remaining)

        _last_historical_request = time.monotonic()


# ==================================================
# PARSE HISTORICAL CANDLES
# ==================================================

def _parse_candles(response):
    """
    Parse Kotak Neo historical candles.

    Supported success responses:
        status = success
        stat = Ok

    Candle layout:
        timestamp, open, high, low, close, volume, open_interest
    """

    columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    if not isinstance(response, dict):
        raise RuntimeError(
            "Unexpected Kotak response type: "
            f"{type(response).__name__}"
        )

    if _is_rate_limit_error(response):
        retry_after = _get_retry_after_seconds(response)

        raise KotakRateLimitError(
            "Kotak Neo rejected the historical request with HTTP 429. "
            "Please wait before requesting more data.",
            retry_after=retry_after,
        )

    status = str(
        response.get("status", "")
    ).strip().lower()

    stat = str(
        response.get("stat", "")
    ).strip().lower()

    success = (
        status in {"success", "ok"}
        or stat in {"ok", "success"}
    )

    if not success:

        raise RuntimeError(
            "Kotak Neo historical API error: "
            + _extract_api_error(response)
        )

    data = response.get("data", {})

    if not isinstance(data, dict):
        raise RuntimeError(
            "Kotak returned an unexpected historical data structure."
        )

    candles = data.get("candles", [])

    if not isinstance(candles, list):
        raise RuntimeError(
            "Kotak historical candles are not in list format."
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
            "Kotak returned candles, but the OHLCV format "
            "could not be recognised."
        )

    df = pd.DataFrame(records)

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
        "volume",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(subset=columns)

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
    instruments_file=INSTRUMENTS_FILE,
):
    """
    Fetch daily historical OHLCV data from Kotak Neo.

    Maximum request period:
        180 calendar days per request.

    Requests are spaced using HISTORICAL_REQUEST_GAP.

    IMPORTANT:
        On HTTP 429, this function stops immediately.
        It does not automatically retry and create more API traffic.
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
        instruments_file=instruments_file,
    )

    all_data = []

    current_start = start_date

    while current_start <= end_date:

        # 180 calendar days inclusive.
        current_end = min(
            current_start + timedelta(days=179),
            end_date,
        )

        print(
            f"Downloading {symbol}: "
            f"{current_start} to {current_end}"
        )

        # Space requests apart.
        _wait_for_request_slot()

        try:

            response = client.historical_data(
                neosymbol=neo_symbol,
                interval="D",
                from_date=current_start.isoformat(),
                to_date=current_end.isoformat(),
            )

        except Exception as exc:

            if _is_rate_limit_error(exc):

                retry_after = _get_retry_after_seconds(exc)

                if retry_after is not None:
                    guidance = (
                        f"Kotak supplied Retry-After: "
                        f"{retry_after} seconds."
                    )
                else:
                    guidance = (
                        "Kotak did not expose a Retry-After value. "
                        "Wait several minutes before trying again."
                    )

                raise KotakRateLimitError(
                    f"Kotak Neo rate limit reached while downloading "
                    f"{symbol} ({current_start} to {current_end}). "
                    f"{guidance} No further requests were sent. "
                    f"Original error: {exc}",
                    retry_after=retry_after,
                ) from exc

            raise RuntimeError(
                f"Kotak historical request failed for {symbol} "
                f"from {current_start} to {current_end}: {exc}"
            ) from exc

        # Some SDK versions return an error dictionary instead
        # of raising an exception.
        if _is_rate_limit_error(response):

            retry_after = _get_retry_after_seconds(response)

            if retry_after is not None:
                guidance = (
                    f"Retry-After: {retry_after} seconds."
                )
            else:
                guidance = (
                    "No Retry-After value was supplied. "
                    "Wait several minutes before trying again."
                )

            raise KotakRateLimitError(
                f"Kotak Neo returned HTTP 429 for {symbol} "
                f"({current_start} to {current_end}). "
                f"{guidance} No further requests were sent.",
                retry_after=retry_after,
            )

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
        ignore_index=True,
    )

    result = (
        result.sort_values("date")
        .drop_duplicates(
            subset="date",
            keep="last",
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
    output_dir=None,
):
    """Save historical data locally as a CSV file."""

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
        exist_ok=True,
    )

    output_file = (
        output_dir / f"{symbol}_daily.csv"
    )

    price_df.to_csv(
        output_file,
        index=False,
    )

    return output_file
