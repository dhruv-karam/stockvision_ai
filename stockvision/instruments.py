
import pandas as pd
from pathlib import Path

from stockvision.data import (
    create_neo_client,
    INSTRUMENTS_FILE,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

COLUMNS = [
    "symbol",
    "exchange_segment",
    "instrument_token",
]


# --------------------------------------------------
# FIND NSE EQUITY
# --------------------------------------------------

def find_nse_equity(symbol, client):
    """
    Search Kotak Neo for an exact NSE cash-market equity.

    Returns:
        {
            "symbol": "RELIANCE",
            "exchange_segment": "nse_cm",
            "instrument_token": "verified_token"
        }
    """

    symbol = str(symbol).strip().upper()

    if not symbol:
        raise ValueError("Stock symbol cannot be empty.")

    try:
        results = client.search_scrip(
            exchange_segment="nse_cm",
            symbol=symbol,
            expiry="",
            option_type="",
            strike_price="",
        )

    except Exception as exc:
        raise RuntimeError(
            f"Kotak Neo search request failed for {symbol}: {exc}"
        ) from exc

    # Handle API error responses.
    if isinstance(results, dict):

        error = (
            results.get("error")
            or results.get("Error")
            or results.get("Error Message")
            or results.get("message")
        )

        if error:
            raise RuntimeError(
                f"Kotak Neo instrument search error: {error}"
            )

        raise RuntimeError(
            f"Unexpected API response for {symbol}: {results}"
        )

    if not isinstance(results, list):
        raise RuntimeError(
            f"Expected a list of instruments for {symbol}, "
            f"received {type(results).__name__}."
        )

    # Filter exact NSE cash-market equity records.
    matches = []

    for item in results:

        if not isinstance(item, dict):
            continue

        returned_symbol = str(
            item.get("pSymbolName") or ""
        ).strip().upper()

        exchange = str(
            item.get("pExchSeg") or ""
        ).strip().lower()

        group = str(
            item.get("pGroup") or ""
        ).strip().upper()

        trading_symbol = str(
            item.get("pTrdSymbol") or ""
        ).strip().upper()

        is_equity = (
            returned_symbol == symbol
            and exchange == "nse_cm"
            and group == "EQ"
            and trading_symbol.endswith("-EQ")
        )

        if is_equity:
            matches.append(item)

    if not matches:
        raise ValueError(
            f"No exact NSE equity instrument found for {symbol}. "
            "Check the symbol or its listing status."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple NSE equity instruments found for {symbol}. "
            "Manual verification is required."
        )

    instrument = matches[0]

    token = instrument.get("pSymbol")

    if pd.isna(token):
        raise ValueError(
            f"Instrument token is missing for {symbol}."
        )

    token = str(token).strip()

    if not token.isdigit() or int(token) <= 0:
        raise ValueError(
            f"Invalid instrument token returned for {symbol}: {token}"
        )

    return {
        "symbol": symbol,
        "exchange_segment": "nse_cm",
        "instrument_token": token,
    }


# --------------------------------------------------
# LOAD EXISTING INSTRUMENTS
# --------------------------------------------------

def load_instruments(output_file=INSTRUMENTS_FILE):
    """
    Load existing instrument records safely.
    """

    output_file = Path(output_file)

    if not output_file.exists():
        return pd.DataFrame(columns=COLUMNS)

    try:
        existing = pd.read_csv(
            output_file,
            dtype=str
        )

    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLUMNS)

    except Exception as exc:
        raise RuntimeError(
            f"Could not read instrument file: {exc}"
        ) from exc

    for column in COLUMNS:
        if column not in existing.columns:
            existing[column] = ""

    existing = existing[COLUMNS].copy()

    existing["symbol"] = (
        existing["symbol"]
        .fillna("")
        .str.strip()
        .str.upper()
    )

    return existing[
        existing["symbol"] != ""
    ].reset_index(drop=True)


# --------------------------------------------------
# UPDATE INSTRUMENT MASTER
# --------------------------------------------------

def update_instruments(
    symbols,
    client=None,
    output_file=INSTRUMENTS_FILE,
):
    """
    Search NSE equity instruments and update instruments.csv.

    Existing instruments are retained.

    Successfully found instruments are added or replaced.
    Failed lookups do not delete previously saved records.

    Returns:
        {
            "updated_file": "...",
            "successful": [...],
            "failed": [...]
        }
    """

    if isinstance(symbols, str):
        symbols = [symbols]

    if symbols is None:
        raise ValueError("Provide at least one stock symbol.")

    # Normalize symbols and remove duplicates.
    normalized_symbols = []

    for symbol in symbols:

        if symbol is None:
            continue

        if pd.isna(symbol):
            continue

        symbol = str(symbol).strip().upper()

        if symbol and symbol not in normalized_symbols:
            normalized_symbols.append(symbol)

    if not normalized_symbols:
        raise ValueError(
            "No valid stock symbols were provided."
        )

    # Create Kotak client only when one was not supplied.
    if client is None:
        client = create_neo_client()

    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Read existing instrument records.
    existing = load_instruments(output_file)

    existing_records = {}

    for _, row in existing.iterrows():

        existing_symbol = str(
            row["symbol"]
        ).strip().upper()

        if not existing_symbol:
            continue

        existing_records[existing_symbol] = {
            "symbol": existing_symbol,
            "exchange_segment": (
                str(row["exchange_segment"]).strip()
                if pd.notna(row["exchange_segment"])
                else ""
            ),
            "instrument_token": (
                str(row["instrument_token"]).strip()
                if pd.notna(row["instrument_token"])
                else ""
            ),
        }

    successful = []
    failed = []

    # Perform individual lookups.
    for symbol in normalized_symbols:

        try:

            instrument = find_nse_equity(
                symbol=symbol,
                client=client,
            )

            existing_records[symbol] = instrument

            successful.append(instrument)

        except Exception as error:

            failed.append({
                "symbol": symbol,
                "error": str(error),
            })

    # Build updated instrument master.
    updated = pd.DataFrame(
        list(existing_records.values()),
        columns=COLUMNS,
    )

    if not updated.empty:

        updated = (
            updated.sort_values("symbol")
            .drop_duplicates(
                subset="symbol",
                keep="last"
            )
            .reset_index(drop=True)
        )

    # Save the instrument master.
    updated.to_csv(
        output_file,
        index=False,
    )

    return {
        "updated_file": str(output_file),
        "successful": successful,
        "failed": failed,
    }

