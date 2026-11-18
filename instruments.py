import pandas as pd
from pathlib import Path

from stockvision.data import (
    create_neo_client,
    INSTRUMENTS_FILE,
)


def find_nse_equity(symbol, client):
    """
    Search Kotak Neo for an NSE equity instrument.

    Returns:
        symbol, exchange_segment, instrument_token
    """

    symbol = str(symbol).strip().upper()

    if not symbol:
        raise ValueError("Stock symbol cannot be empty.")

    results = client.search_scrip(
        exchange_segment="nse_cm",
        symbol=symbol,
        expiry="",
        option_type="",
        strike_price="",
    )

    if isinstance(results, dict) and results.get("error"):
        raise RuntimeError(
            f"Kotak Neo instrument search failed: {results['error']}"
        )

    if not isinstance(results, list):
        raise RuntimeError(
            f"Unexpected instrument search response for {symbol}."
        )

    # Keep only exact NSE equity matches.
    matches = []

    for item in results:

        if not isinstance(item, dict):
            continue

        returned_symbol = str(
            item.get("pSymbolName", "")
        ).strip().upper()

        exchange = str(
            item.get("pExchSeg", "")
        ).strip().lower()

        group = str(
            item.get("pGroup", "")
        ).strip().upper()

        trading_symbol = str(
            item.get("pTrdSymbol", "")
        ).strip().upper()

        if (
            returned_symbol == symbol
            and exchange == "nse_cm"
            and group == "EQ"
            and trading_symbol.endswith("-EQ")
        ):
            matches.append(item)

    if not matches:
        raise ValueError(
            f"No NSE equity instrument found for {symbol}. "
            "Check the spelling or whether the stock is listed."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple NSE equity instruments found for {symbol}. "
            "Manual verification is required."
        )

    instrument = matches[0]

    token = instrument.get("pSymbol")

    if token is None or str(token).strip() == "":
        raise ValueError(
            f"Instrument token missing for {symbol}."
        )

    return {
        "symbol": symbol,
        "exchange_segment": "nse_cm",
        "instrument_token": str(token),
    }


def update_instruments(
    symbols,
    client=None,
    output_file=INSTRUMENTS_FILE,
):
    """
    Look up NSE equity tokens and update instruments.csv.

    Existing instruments are retained. Newly found instruments
    are added or updated.
    """

    if isinstance(symbols, str):
        symbols = [symbols]

    symbols = list(dict.fromkeys(
        str(symbol).strip().upper()
        for symbol in symbols
        if str(symbol).strip()
    ))

    if not symbols:
        raise ValueError("Provide at least one stock symbol.")

    if client is None:
        client = create_neo_client()

    output_file = Path(output_file)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    columns = [
        "symbol",
        "exchange_segment",
        "instrument_token",
    ]

    if output_file.exists():
        existing = pd.read_csv(
            output_file,
            dtype=str
        )
    else:
        existing = pd.DataFrame(columns=columns)

    for column in columns:
        if column not in existing.columns:
            existing[column] = ""

    existing = existing[columns]

    existing_records = {
        row["symbol"].strip().upper(): row.to_dict()
        for _, row in existing.iterrows()
        if pd.notna(row["symbol"]) and str(row["symbol"]).strip()
    }

    successful = []
    failed = []

    for symbol in symbols:
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

    updated = pd.DataFrame(
        list(existing_records.values()),
        columns=columns,
    )

    updated = updated.sort_values(
        "symbol"
    ).reset_index(drop=True)

    updated.to_csv(
        output_file,
        index=False,
    )

    return {
        "updated_file": str(output_file),
        "successful": successful,
        "failed": failed,
    }
