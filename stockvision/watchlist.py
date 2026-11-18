import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WATCHLIST_FILE = PROJECT_ROOT / "data" / "watchlist.csv"

COLUMNS = ["symbol", "added_at"]


def validate_symbol(symbol):
    """Validate and standardize an NSE stock symbol."""

    symbol = str(symbol).strip().upper()

    if not symbol:
        raise ValueError("Stock symbol cannot be empty.")

    if not re.fullmatch(r"[A-Z0-9&.-]{1,30}", symbol):
        raise ValueError(
            f"Invalid stock symbol: {symbol}"
        )

    return symbol


def load_watchlist(file_path=WATCHLIST_FILE):
    """Load saved watchlist symbols."""

    file_path = Path(file_path)

    if not file_path.exists():
        return pd.DataFrame(columns=COLUMNS)

    try:
        df = pd.read_csv(file_path, dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=COLUMNS)

    for column in COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df = df[COLUMNS].dropna(subset=["symbol"]).copy()

    df["symbol"] = (
        df["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df = df[df["symbol"] != ""]
    df = df.drop_duplicates(subset=["symbol"], keep="last")

    return df.reset_index(drop=True)


def save_watchlist(df, file_path=WATCHLIST_FILE):
    """Save watchlist data to CSV."""

    file_path = Path(file_path)

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_df = df[COLUMNS].copy()

    clean_df.to_csv(
        file_path,
        index=False,
    )


def add_symbol(symbol, file_path=WATCHLIST_FILE):
    """Add a stock symbol to the watchlist."""

    symbol = validate_symbol(symbol)

    watchlist = load_watchlist(file_path)

    if symbol in watchlist["symbol"].tolist():
        return {
            "success": False,
            "message": f"{symbol} is already in your watchlist.",
        }

    new_entry = pd.DataFrame([{
        "symbol": symbol,
        "added_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }])

    watchlist = pd.concat(
        [watchlist, new_entry],
        ignore_index=True,
    )

    save_watchlist(watchlist, file_path)

    return {
        "success": True,
        "message": f"{symbol} added to your watchlist.",
    }


def remove_symbol(symbol, file_path=WATCHLIST_FILE):
    """Remove a stock symbol from the watchlist."""

    symbol = validate_symbol(symbol)

    watchlist = load_watchlist(file_path)

    if symbol not in watchlist["symbol"].tolist():
        return {
            "success": False,
            "message": f"{symbol} was not found in your watchlist.",
        }

    watchlist = watchlist[
        watchlist["symbol"] != symbol
    ].reset_index(drop=True)

    save_watchlist(watchlist, file_path)

    return {
        "success": True,
        "message": f"{symbol} removed from your watchlist.",
    }
