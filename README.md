# StockVision AI

An experimental Indian stock-market analysis and forecasting application
built with Python, Streamlit, XGBoost and the Kotak Neo Trade API.

## Features

- Fetch daily historical stock data through Kotak Neo.
- Upload historical OHLCV data through CSV.
- Calculate technical indicators:
  - RSI
  - MACD
  - SMA and EMA
  - Bollinger Bands
  - ATR
  - Historical volatility
  - Volume indicators
- Train XGBoost forecasting models for:
  - 21 trading days
  - 63 trading days
  - 126 trading days
  - 252 trading days
- Evaluate models using chronological validation.
- Explore 3-year and 5-year EPS growth and P/E scenarios.
- View interactive charts and download historical data.

## Project Structure

```text
stockvision_ai/
├── app.py
├── test_kotak.py
├── requirements.txt
├── README.md
├── .env
├── .env.example
├── .gitignore
├── instruments.csv
├── data/
├── models/
└── stockvision/
    ├── __init__.py
    ├── data.py
    ├── features.py
    ├── modeling.py
    └── valuation.py
