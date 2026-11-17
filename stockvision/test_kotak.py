import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from stockvision.data import fetch_daily_history


def main():
    parser = argparse.ArgumentParser(
        description="Test Kotak Neo historical market data."
    )

    parser.add_argument(
        "--symbol",
        default="RELIANCE",
        help="Stock symbol listed in instruments.csv"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of calendar days of history to fetch"
    )

    args = parser.parse_args()

    if args.days <= 0:
        print("Error: --days must be a positive number.")
        sys.exit(1)

    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    print("StockVision AI - Kotak Neo Connection Test")
    print("-" * 45)
    print(f"Symbol: {args.symbol.upper()}")
    print(f"From:   {start_date}")
    print(f"To:     {end_date}")
    print()

    try:
        df = fetch_daily_history(
            symbol=args.symbol,
            start_date=start_date,
            end_date=end_date
        )

        if df.empty:
            print(
                "The API call completed, but no candle data was returned."
            )
            print(
                "Check the stock symbol, instrument token, date range "
                "and market-data availability."
            )
            sys.exit(1)

        output_dir = Path("data")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = (
            output_dir / f"{args.symbol.upper()}_daily.csv"
        )

        df.to_csv(output_file, index=False)

        print("Historical data fetched successfully!")
        print(f"Rows received: {len(df)}")
        print(f"Saved to: {output_file}")
        print()
        print("Most recent records:")
        print(df.tail(5).to_string(index=False))

    except Exception as error:
        print("\nKotak Neo test failed.")
        print(f"Error: {error}")
        print(
            "\nCheck that your .env file, SDK installation, "
            "instrument token and API access are configured correctly."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
