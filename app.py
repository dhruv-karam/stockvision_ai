
import os
import re
import hashlib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# IMPORTS
# ============================================================

from stockvision.data import (
    create_neo_client,
    fetch_daily_history,
)

from stockvision.features import build_features

from stockvision.instruments import update_instruments

from stockvision.fundamentals import (
    fetch_auto_fundamentals,
)

from stockvision.signals import (
    generate_technical_signal,
)

from stockvision.risk import (
    calculate_risk_metrics,
)

from stockvision.watchlist import (
    load_watchlist,
    add_symbol,
    remove_symbol,
)

from stockvision.scanner import (
    scan_watchlist,
)


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="StockVision AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

LONG_TERM_YEARS = 5
REQUIRED_RETURN = 0.15

INDIA_TODAY = datetime.now(
    ZoneInfo("Asia/Kolkata")
).date()

PERIOD_OPTIONS = {
    "1 Year": 365,
    "3 Years": 1095,
    "5 Years": 1825,
}


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1450px;
        padding-top: 1.3rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(125, 135, 150, 0.07);
        border: 1px solid rgba(150, 150, 150, 0.16);
        border-radius: 12px;
        padding: 13px;
    }

    h1 {
        letter-spacing: -0.04em;
    }

    div[data-testid="stExpander"] {
        border-radius: 10px;
    }

    div[data-testid="stForm"] {
        border: none;
        padding: 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# STREAMLIT CLOUD SECRETS
# ============================================================

def load_kotak_credentials():

    try:
        key = st.secrets.get(
            "NEO_CONSUMER_KEY",
            "",
        )

        environment = st.secrets.get(
            "NEO_ENVIRONMENT",
            "prod",
        )

    except Exception:
        key = ""
        environment = "prod"

    if key:
        os.environ["NEO_CONSUMER_KEY"] = str(key).strip()

    if environment:
        os.environ["NEO_ENVIRONMENT"] = str(environment).strip()

    actual_key = os.getenv(
        "NEO_CONSUMER_KEY",
        "",
    ).strip()

    return bool(
        actual_key
        and not actual_key.startswith("YOUR_")
    )


HAS_KOTAK_KEY = load_kotak_credentials()

CREDENTIAL_HASH = (
    hashlib.sha256(
        os.getenv("NEO_CONSUMER_KEY", "").encode()
    ).hexdigest()[:16]
    if HAS_KOTAK_KEY
    else "no-key"
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "selected_symbol": "RELIANCE",
    "selected_company_name": "Reliance Industries Limited",

    "price_data": None,
    "active_symbol": None,
    "data_source": None,
    "data_error": None,
    "fetch_signature": None,

    "scanner_results": None,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


def clear_market_state():

    for key in (
        "price_data",
        "active_symbol",
        "data_source",
        "data_error",
        "fetch_signature",
        "scanner_results",
    ):
        st.session_state[key] = None


# ============================================================
# LIVE NSE COMPANY SEARCH
# ============================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def search_nse_companies(query):
    """
    Search Yahoo Finance for NSE-listed companies.

    Returns:
        symbol, company_name, exchange, sector-independent profile

    Only NSE equity symbols ending in .NS are returned.
    """

    query = str(query).strip()

    columns = [
        "symbol",
        "company_name",
        "exchange",
    ]

    if len(query) < 2:
        return pd.DataFrame(columns=columns)

    try:
        import yfinance as yf

        search = yf.Search(
            query,
            max_results=15,
            news_count=0,
            lists_count=0,
            include_cb=False,
            include_nav_links=False,
            enable_fuzzy_query=True,
            timeout=8,
            raise_errors=False,
        )

        quotes = search.quotes or []

    except Exception:
        return pd.DataFrame(columns=columns)

    records = []

    for quote in quotes:

        if not isinstance(quote, dict):
            continue

        ticker = str(
            quote.get("symbol", "")
        ).strip().upper()

        quote_type = str(
            quote.get("quoteType", "")
        ).strip().upper()

        # Only NSE-listed equity shares.
        if not ticker.endswith(".NS"):
            continue

        if quote_type and quote_type != "EQUITY":
            continue

        company_name = (
            quote.get("longname")
            or quote.get("shortname")
            or quote.get("longName")
            or ticker.replace(".NS", "")
        )

        records.append({
            "symbol": ticker.replace(".NS", ""),
            "company_name": str(company_name),
            "exchange": "NSE",
        })

    if not records:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(records)

    result = (
        result.drop_duplicates(subset="symbol")
        .reset_index(drop=True)
    )

    return result[columns]


# ============================================================
# CACHED MARKET DATA
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def get_market_data(
    symbol,
    start_date,
    end_date,
    credential_hash,
):

    _ = credential_hash

    client = create_neo_client()

    lookup = update_instruments(
        [symbol],
        client=client,
    )

    if not lookup["successful"]:

        errors = "; ".join(
            item["error"]
            for item in lookup["failed"]
        )

        raise RuntimeError(
            f"NSE instrument lookup failed: {errors}"
        )

    prices = fetch_daily_history(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        client=client,
    )

    if prices.empty:
        raise RuntimeError(
            f"No historical prices received for {symbol}."
        )

    return prices


# ============================================================
# CACHED COMPANY FUNDAMENTALS
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False,
)
def get_fundamentals(
    symbol,
    market_price,
):

    return fetch_auto_fundamentals(
        symbol=symbol,
        market_price=market_price,
        forecast_years=LONG_TERM_YEARS,
    )


# ============================================================
# FORMATTING
# ============================================================

def money(value):

    if value is None or pd.isna(value):
        return "N/A"

    return f"₹{float(value):,.2f}"


def percentage(value):

    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):+.2f}%"


def multiple(value):

    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):.2f}x"


# ============================================================
# SHORT-TERM PRICE LEVELS
# ============================================================

def calculate_short_term_levels(
    price_df,
    feature_df,
):
    """
    Calculate indicative 21-trading-day technical levels.

    Fair entry:
        Midpoint between 20-day support and SMA 20.

    Target 1:
        Entry + 1.5 ATR.

    Target 2:
        Entry + 3 ATR.

    Stop:
        Entry - 1.5 ATR.

    These are technical reference levels, not calibrated
    probability-based predictions.
    """

    if len(feature_df) < 20:
        raise ValueError(
            "At least 20 trading observations are required."
        )

    latest = feature_df.iloc[-1]

    sma20 = latest.get("sma_20")
    atr = latest.get("atr_14")

    if pd.isna(sma20) or pd.isna(atr):
        raise ValueError(
            "SMA 20 or ATR 14 is unavailable."
        )

    if float(sma20) <= 0 or float(atr) <= 0:
        raise ValueError(
            "Invalid moving average or ATR."
        )

    recent_low = pd.to_numeric(
        price_df.tail(20)["low"],
        errors="coerce",
    )

    support = float(
        recent_low.min()
    )

    if not np.isfinite(support) or support <= 0:
        raise ValueError(
            "Recent 20-day support is unavailable."
        )

    current_price = float(
        latest["close"]
    )

    entry_low = min(
        support,
        float(sma20),
    )

    entry_high = max(
        support,
        float(sma20),
    )

    fair_entry = (
        entry_low + entry_high
    ) / 2

    atr_value = float(atr)

    target1 = fair_entry + (1.5 * atr_value)

    target2 = fair_entry + (3.0 * atr_value)

    stop_loss = fair_entry - (1.5 * atr_value)

    if stop_loss <= 0:
        raise ValueError(
            "Stop level is outside the valid price range."
        )

    if current_price > entry_high:
        status = "Above preferred entry zone"

    elif current_price >= entry_low:
        status = "Inside preferred entry zone"

    else:
        status = "Below recent support"

    return {
        "current_price": current_price,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "fair_entry": fair_entry,
        "target1": target1,
        "target2": target2,
        "stop_loss": stop_loss,
        "entry_status": status,
    }


# ============================================================
# LONG-TERM PRICE LEVELS AND EXPECTED CAGR
# ============================================================

def calculate_long_term_levels(
    fundamental_data,
    current_price,
):

    scenarios = fundamental_data["scenarios"]

    bear = scenarios[
        scenarios["scenario"] == "Bear"
    ]

    base = scenarios[
        scenarios["scenario"] == "Base"
    ]

    bull = scenarios[
        scenarios["scenario"] == "Bull"
    ]

    if base.empty:
        raise ValueError(
            "Base valuation scenario is unavailable."
        )

    bear_target = (
        float(bear.iloc[0]["estimated_future_price"])
        if not bear.empty
        else np.nan
    )

    base_target = float(
        base.iloc[0]["estimated_future_price"]
    )

    bull_target = (
        float(bull.iloc[0]["estimated_future_price"])
        if not bull.empty
        else np.nan
    )

    if (
        current_price <= 0
        or base_target <= 0
        or not np.isfinite(base_target)
    ):
        raise ValueError(
            "Invalid long-term valuation inputs."
        )

    fair_entry = (
        base_target
        / ((1 + REQUIRED_RETURN) ** LONG_TERM_YEARS)
    )

    expected_cagr = (
        (base_target / current_price)
        ** (1 / LONG_TERM_YEARS)
        - 1
    ) * 100

    return {
        "fair_entry": fair_entry,
        "bear_target": bear_target,
        "base_target": base_target,
        "bull_target": bull_target,
        "expected_cagr_pct": expected_cagr,
        "required_return_pct": REQUIRED_RETURN * 100,
        "years": LONG_TERM_YEARS,
    }


# ============================================================
# HEADER
# ============================================================

st.title("📈 StockVision AI")

st.caption(
    "Indian Equity Research · Fair Entry Prices · "
    "Price Targets · Expected CAGR"
)


# ============================================================
# SEARCH BAR WITH COMPANY SUGGESTIONS
# ============================================================

st.subheader("Find a Company")

search_query = st.text_input(
    "Search by company name or NSE ticker",
    placeholder="Type Reliance, Tata, Infosys, HDFC...",
    key="company_search_query",
    label_visibility="collapsed",
)

search_results = search_nse_companies(
    search_query
)

selected_symbol = st.session_state["selected_symbol"]

if not search_results.empty:

    # Construct human-readable suggestion labels.
    labels = {
        row["symbol"]: (
            f"{row['company_name']}  ·  "
            f"{row['symbol']}  ·  NSE"
        )
        for _, row in search_results.iterrows()
    }

    options = search_results["symbol"].tolist()

    matching_index = (
        options.index(selected_symbol)
        if selected_symbol in options
        else None
    )

    selected_result = st.selectbox(
        "Matching NSE companies",
        options=options,
        index=matching_index,
        format_func=lambda ticker: labels.get(
            ticker,
            ticker,
        ),
        placeholder="Select a company from the suggestions",
        key=(
            "company_suggestion_"
            + hashlib.sha1(
                search_query.lower().encode()
            ).hexdigest()[:12]
        ),
        label_visibility="collapsed",
    )

    if selected_result is not None:

        if selected_result != selected_symbol:

            selected_row = search_results[
                search_results["symbol"] == selected_result
            ].iloc[0]

            st.session_state["selected_symbol"] = selected_result

            st.session_state["selected_company_name"] = (
                selected_row["company_name"]
            )

            clear_market_state()

            st.rerun()

else:

    if len(search_query.strip()) >= 2:

        st.info(
            "No NSE equity matches found. Try another company name or ticker."
        )

    else:

        st.caption(
            f"Currently selected: {selected_symbol}"
        )


symbol = st.session_state["selected_symbol"]


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

with st.sidebar:

    st.title("Market Settings")

    period_label = st.selectbox(
        "Historical data range",
        list(PERIOD_OPTIONS.keys()),
        index=0,
    )

    period_days = PERIOD_OPTIONS[period_label]

    refresh = st.button(
        "↻ Refresh market data",
        use_container_width=True,
    )

    with st.expander("Upload historical CSV"):

        uploaded_file = st.file_uploader(
            "OHLCV CSV",
            type=["csv"],
        )

        if st.button("Load CSV"):

            if uploaded_file is None:

                st.warning(
                    "Select a CSV file first."
                )

            else:

                try:

                    uploaded = pd.read_csv(
                        uploaded_file
                    )

                    required = [
                        "date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]

                    missing = [
                        c for c in required
                        if c not in uploaded.columns
                    ]

                    if missing:

                        st.error(
                            "Missing columns: "
                            + ", ".join(missing)
                        )

                    else:

                        st.session_state["price_data"] = (
                            uploaded[required].copy()
                        )

                        st.session_state["active_symbol"] = symbol

                        st.session_state["data_source"] = "CSV Upload"

                        st.session_state["data_error"] = None

                        st.session_state["fetch_signature"] = (
                            f"CSV|{symbol}"
                        )

                        st.success(
                            "Historical data loaded."
                        )

                        st.rerun()

                except Exception as exc:

                    st.error(
                        f"CSV error: {exc}"
                    )


# ============================================================
# AUTOMATIC MARKET DATA FETCH
# ============================================================

today = INDIA_TODAY

start_date = today - timedelta(
    days=period_days
)

end_date = today

request_signature = (
    f"{symbol}|{period_days}|{CREDENTIAL_HASH}"
)

if refresh:

    get_market_data.clear()

    st.session_state["fetch_signature"] = None

    st.session_state["data_source"] = None

if (
    st.session_state["fetch_signature"] != request_signature
    and st.session_state["data_source"] != "CSV Upload"
):

    st.session_state["fetch_signature"] = request_signature

    st.session_state["price_data"] = None

    st.session_state["active_symbol"] = None

    st.session_state["data_source"] = None

    st.session_state["data_error"] = None

    if HAS_KOTAK_KEY:

        try:

            with st.spinner(
                f"Loading {symbol} historical data..."
            ):

                prices = get_market_data(
                    symbol,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    CREDENTIAL_HASH,
                )

            st.session_state["price_data"] = prices

            st.session_state["active_symbol"] = symbol

            st.session_state["data_source"] = "Kotak Neo"

        except Exception as exc:

            st.session_state["data_error"] = str(exc)

    else:

        st.session_state["data_error"] = (
            "Kotak credentials are not configured. "
            "Add NEO_CONSUMER_KEY in Streamlit Secrets."
        )


# ============================================================
# CHECK DATA
# ============================================================

price_df = st.session_state["price_data"]

active_symbol = (
    st.session_state["active_symbol"]
    or symbol
)

if price_df is None:

    st.warning(
        st.session_state["data_error"]
        or "Historical stock data is unavailable."
    )

    st.info(
        "Select an NSE company and configure Kotak Neo credentials. "
        "Historical prices are fetched automatically."
    )

    st.stop()


# ============================================================
# CLEAN HISTORICAL DATA
# ============================================================

required_columns = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

price_df = price_df.copy()

price_df["date"] = pd.to_datetime(
    price_df["date"],
    errors="coerce",
)

for column in [
    "open",
    "high",
    "low",
    "close",
    "volume",
]:

    price_df[column] = pd.to_numeric(
        price_df[column],
        errors="coerce",
    )

price_df = (
    price_df.dropna(subset=required_columns)
    .sort_values("date")
    .drop_duplicates("date", keep="last")
    .reset_index(drop=True)
)

if price_df.empty:

    st.error(
        "No valid price observations are available."
    )

    st.stop()


# ============================================================
# FEATURE ENGINEERING
# ============================================================

try:

    feature_df = build_features(
        price_df
    )

except Exception as exc:

    st.error(
        f"Technical indicator calculation failed: {exc}"
    )

    st.stop()

if feature_df.empty:

    st.error(
        "No valid technical indicators could be calculated."
    )

    st.stop()


latest = feature_df.iloc[-1]

current_price = float(
    latest["close"]
)

previous_price = (
    float(feature_df["close"].iloc[-2])
    if len(feature_df) > 1
    else current_price
)

daily_change = (
    (current_price / previous_price - 1) * 100
    if previous_price > 0
    else 0
)

latest_date = pd.to_datetime(
    latest["date"]
).date()


# ============================================================
# COMPANY PROFILE
# ============================================================

company_name = st.session_state.get(
    "selected_company_name",
    active_symbol,
)

sector = "Not available"
industry = "Not available"

try:

    import yfinance as yf

    company_ticker = yf.Ticker(
        f"{active_symbol}.NS"
    )

    company_info = company_ticker.get_info()

    company_name = (
        company_info.get("longName")
        or company_info.get("shortName")
        or company_name
    )

    sector = company_info.get(
        "sector",
        "Not available",
    )

    industry = company_info.get(
        "industry",
        "Not available",
    )

except Exception:
    pass


# ============================================================
# COMPANY HEADER
# ============================================================

st.header(company_name)

st.caption(
    f"NSE: {active_symbol} · {sector} · {industry}"
)

st.caption(
    f"Data: {st.session_state['data_source']} · "
    f"Latest available candle: {latest_date:%d %b %Y} · "
    f"{len(feature_df):,} daily records"
)


# ============================================================
# CURRENT MARKET SUMMARY
# ============================================================

a, b, c, d = st.columns(4)

a.metric(
    "Current Price",
    money(current_price),
    percentage(daily_change),
)

rsi = latest.get("rsi_14", np.nan)

b.metric(
    "RSI (14)",
    f"{rsi:.2f}" if pd.notna(rsi) else "N/A",
)

c.metric(
    "Daily Volume",
    f"{float(latest['volume']):,.0f}",
)

d.metric(
    "Historical Records",
    f"{len(feature_df):,}",
)

st.divider()


# ============================================================
# AUTOMATIC SHORT-TERM LEVELS
# ============================================================

short_levels = None
short_error = None

try:

    short_levels = calculate_short_term_levels(
        price_df,
        feature_df,
    )

except Exception as exc:

    short_error = str(exc)


# ============================================================
# AUTOMATIC LONG-TERM FUNDAMENTALS
# ============================================================

fundamentals = None
long_levels = None
fundamental_error = None

try:

    with st.spinner(
        f"Calculating {active_symbol} fundamental valuation..."
    ):

        fundamentals = get_fundamentals(
            active_symbol,
            current_price,
        )

    long_levels = calculate_long_term_levels(
        fundamentals,
        current_price,
    )

except Exception as exc:

    fundamental_error = str(exc)


# ============================================================
# MAIN INVESTMENT DASHBOARD
# ============================================================

st.header("🎯 Fair Entry Prices & Targets")

st.caption(
    "Automatically calculated short-term technical levels "
    "and long-term fundamental valuation."
)

short_col, long_col = st.columns(2)


# ============================================================
# SHORT-TERM CARD
# ============================================================

with short_col:

    st.subheader("⚡ Short Term")

    st.caption(
        "21 Trading Days · Technical Scenario"
    )

    if short_levels is not None:

        st.metric(
            "Indicative Fair Entry",
            money(short_levels["fair_entry"]),
            short_levels["entry_status"],
        )

        st.write(
            "**Preferred Entry Zone**"
        )

        st.write(
            f"{money(short_levels['entry_low'])} – "
            f"{money(short_levels['entry_high'])}"
        )

        a, b = st.columns(2)

        a.metric(
            "Target 1",
            money(short_levels["target1"]),
        )

        b.metric(
            "Target 2",
            money(short_levels["target2"]),
        )

        st.metric(
            "Stop-Loss Reference",
            money(short_levels["stop_loss"]),
        )

        st.caption(
            "Based on 20-day support, SMA 20 and ATR 14. "
            "These are technical reference levels, not AI price predictions."
        )

    else:

        st.warning(
            f"Short-term levels unavailable: {short_error}"
        )


# ============================================================
# LONG-TERM CARD
# ============================================================

with long_col:

    st.subheader("🏦 Long Term")

    st.caption(
        "5 Years · Automatic Fundamental Valuation"
    )

    if long_levels is not None:

        st.metric(
            "Indicative Fair Entry",
            money(long_levels["fair_entry"]),
            f"{long_levels['required_return_pct']:.0f}% "
            "annual return hurdle",
        )

        st.metric(
            "Expected Base-Case CAGR",
            percentage(
                long_levels["expected_cagr_pct"]
            ),
            "From current market price",
        )

        a, b, c = st.columns(3)

        a.metric(
            "Bear Target",
            money(long_levels["bear_target"]),
        )

        b.metric(
            "Base Target",
            money(long_levels["base_target"]),
        )

        c.metric(
            "Bull Target",
            money(long_levels["bull_target"]),
        )

    else:

        st.warning(
            "Long-term valuation is unavailable."
        )

        st.caption(
            fundamental_error
        )


# ============================================================
# PRICE LEVEL COMPARISON
# ============================================================

if short_levels is not None or long_levels is not None:

    st.divider()

    st.subheader("Price Level Comparison")

    price_levels = [
        {
            "Level": "Current Price",
            "Price": current_price,
            "Category": "Market",
        }
    ]

    if short_levels is not None:

        price_levels.extend([
            {
                "Level": "Short Entry",
                "Price": short_levels["fair_entry"],
                "Category": "Short Term",
            },
            {
                "Level": "Target 1",
                "Price": short_levels["target1"],
                "Category": "Short Term",
            },
            {
                "Level": "Target 2",
                "Price": short_levels["target2"],
                "Category": "Short Term",
            },
        ])

    if long_levels is not None:

        price_levels.extend([
            {
                "Level": "Long Entry",
                "Price": long_levels["fair_entry"],
                "Category": "Long Term",
            },
            {
                "Level": "Five-Year Base",
                "Price": long_levels["base_target"],
                "Category": "Long Term",
            },
        ])

    comparison = pd.DataFrame(
        price_levels
    )

    chart = px.bar(
        comparison,
        x="Level",
        y="Price",
        color="Category",
        title=f"{active_symbol} Price Levels",
    )

    chart.update_layout(
        height=340,
        xaxis_title="",
        yaxis_title="Price (₹)",
        margin=dict(
            l=5,
            r=5,
            t=45,
            b=5,
        ),
    )

    st.plotly_chart(
        chart,
        use_container_width=True,
    )


# ============================================================
# FUNDAMENTAL HEALTH
# ============================================================

with st.expander(
    "🏢 Company Financial Health",
    expanded=False,
):

    if fundamentals is not None:

        metrics = fundamentals["metrics"]

        st.subheader(
            fundamentals["company"]
        )

        st.caption(
            f"Latest financial year: "
            f"{fundamentals['latest_fiscal_year']}"
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "EPS",
            money(metrics["eps"]),
        )

        b.metric(
            "P/E",
            multiple(metrics["pe_ratio"]),
        )

        c.metric(
            "Revenue CAGR",
            percentage(metrics["revenue_growth_pct"]),
        )

        d.metric(
            "EPS CAGR",
            percentage(metrics["eps_growth_pct"]),
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "ROE",
            percentage(metrics["roe_pct"]),
        )

        b.metric(
            "ROCE",
            percentage(metrics["roce_pct"]),
        )

        c.metric(
            "Operating Margin",
            percentage(metrics["operating_margin_pct"]),
        )

        d.metric(
            "Net Profit Margin",
            percentage(metrics["net_profit_margin_pct"]),
        )

        a, b = st.columns(2)

        a.metric(
            "Debt / Equity",
            multiple(metrics["debt_to_equity"]),
        )

        b.metric(
            "Current Ratio",
            multiple(metrics["current_ratio"]),
        )

        with st.expander("Annual Financial History"):

            st.dataframe(
                fundamentals["financial_history"],
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Valuation Assumptions"):

            st.write(
                fundamentals["assumptions"]
            )

            st.caption(
                f"EPS source: {metrics['eps_source']}"
            )

            st.caption(
                f"Data source: {fundamentals['source']}"
            )

    else:

        st.info(
            "Fundamental data is unavailable for this company. "
            "No financial values are fabricated."
        )

        if fundamental_error:
            st.caption(fundamental_error)


# ============================================================
# TECHNICAL ANALYSIS
# ============================================================

with st.expander(
    "📊 Technical Analysis",
    expanded=False,
):

    chart_columns = [
        c for c in [
            "close",
            "sma_20",
            "sma_50",
            "sma_200",
        ]
        if c == "close"
        or (
            c in feature_df.columns
            and feature_df[c].notna().any()
        )
    ]

    price_chart = px.line(
        feature_df,
        x="date",
        y=chart_columns,
        title=f"{active_symbol} Price Trend",
    )

    price_chart.update_layout(
        height=400,
        hovermode="x unified",
        yaxis_title="Price (₹)",
    )

    st.plotly_chart(
        price_chart,
        use_container_width=True,
    )

    a, b = st.columns(2)

    rsi_chart = px.line(
        feature_df,
        x="date",
        y="rsi_14",
        title="RSI (14)",
    )

    rsi_chart.add_hline(
        y=70,
        line_dash="dash",
    )

    rsi_chart.add_hline(
        y=30,
        line_dash="dash",
    )

    a.plotly_chart(
        rsi_chart,
        use_container_width=True,
    )

    macd_chart = px.line(
        feature_df,
        x="date",
        y=["macd", "macd_signal"],
        title="MACD",
    )

    b.plotly_chart(
        macd_chart,
        use_container_width=True,
    )

    try:

        signal = generate_technical_signal(
            feature_df
        )

        st.subheader("Technical Signal")

        a, b, c = st.columns(3)

        a.metric(
            "Technical Bias",
            signal["overall_signal"],
        )

        b.metric(
            "Technical Score",
            f"{signal['score']:+.1f}/100",
        )

        c.metric(
            "RSI Condition",
            signal["rsi_status"],
        )

        st.dataframe(
            signal["indicators"],
            use_container_width=True,
            hide_index=True,
        )

    except Exception as exc:

        st.info(
            f"Technical signal unavailable: {exc}"
        )


# ============================================================
# HISTORICAL RISK
# ============================================================

with st.expander(
    "⚠️ Historical Risk Analysis",
    expanded=False,
):

    try:

        risk = calculate_risk_metrics(
            feature_df,
            risk_free_rate=0.06,
            confidence_level=0.95,
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "Annual Volatility",
            percentage(risk["annualized_volatility_pct"]),
        )

        b.metric(
            "Maximum Drawdown",
            percentage(risk["maximum_drawdown_pct"]),
        )

        c.metric(
            "Sharpe Ratio",
            (
                f"{risk['sharpe_ratio']:.2f}"
                if pd.notna(risk["sharpe_ratio"])
                else "N/A"
            ),
        )

        d.metric(
            "Daily VaR",
            percentage(risk["daily_var_pct"]),
        )

    except Exception as exc:

        st.warning(
            f"Risk analysis unavailable: {exc}"
        )


# ============================================================
# WATCHLIST
# ============================================================

with st.expander(
    "⭐ Watchlist & Multi-Stock Scanner",
    expanded=False,
):

    with st.form(
        "watchlist_add_form",
        clear_on_submit=True,
    ):

        new_symbol = st.text_input(
            "Add NSE stock",
            placeholder="Example: TCS",
        )

        add_clicked = st.form_submit_button(
            "Add to Watchlist",
        )

    if add_clicked:

        try:

            result = add_symbol(new_symbol)

            if result["success"]:
                st.success(result["message"])
            else:
                st.info(result["message"])

        except Exception as exc:

            st.error(str(exc))

    watchlist = load_watchlist()

    if watchlist.empty:

        st.info(
            "Your watchlist is empty."
        )

    else:

        st.dataframe(
            watchlist,
            use_container_width=True,
            hide_index=True,
        )

        remove_choice = st.selectbox(
            "Remove a stock",
            [""] + watchlist["symbol"].tolist(),
            key="watchlist_remove",
        )

        if st.button("Remove Selected Stock"):

            if remove_choice:

                try:

                    remove_symbol(remove_choice)

                    st.session_state["scanner_results"] = None

                    st.rerun()

                except Exception as exc:

                    st.error(str(exc))

        if st.button(
            "Scan Watchlist",
            type="primary",
        ):

            if not HAS_KOTAK_KEY:

                st.warning(
                    "Kotak API access is required for scanning."
                )

            else:

                try:

                    with st.spinner(
                        "Scanning NSE stocks..."
                    ):

                        client = create_neo_client()

                        update_instruments(
                            watchlist["symbol"].tolist(),
                            client=client,
                        )

                        results = scan_watchlist(
                            watchlist_df=watchlist,
                            lookback_days=500,
                            client=client,
                        )

                    st.session_state["scanner_results"] = results

                except Exception as exc:

                    st.error(
                        f"Scanner failed: {exc}"
                    )

        results = st.session_state["scanner_results"]

        if results is not None:

            st.subheader("Watchlist Results")

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# METHODOLOGY AND DISCLAIMER
# ============================================================

with st.expander(
    "How StockVision calculates price levels",
):

    st.markdown(
        """
        **Short-term entry:** The midpoint between recent 20-day
        price support and the 20-day simple moving average.

        **Short-term targets:** ATR-based technical scenarios
        over approximately 21 trading days.

        **Long-term entry:** The five-year Base-case target
        discounted at a 15% annual required-return hurdle.

        **Expected CAGR:** Annualized return implied by the
        five-year Base target compared with today's market price.

        **Long-term targets:** Derived from historical financial
        growth and Bear/Base/Bull future P/E assumptions.
        """
    )

    st.caption(
        "These calculations are estimates, not guaranteed price "
        "targets or personalized investment advice. Yahoo Finance "
        "data is unofficial and may be incomplete or delayed."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "StockVision AI · Indian Equities Research · "
    "Indicative estimates only."
)
