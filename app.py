
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
# IMPORT PROJECT MODULES
# ============================================================

from stockvision.data import (
    create_neo_client,
    fetch_daily_history,
)

from stockvision.features import build_features

from stockvision.instruments import update_instruments

from stockvision.modeling import (
    train_forecast_models,
    predict_latest,
)

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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="StockVision AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
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
# DASHBOARD DESIGN
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1450px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(125, 135, 150, 0.07);
        border: 1px solid rgba(150, 150, 150, 0.18);
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
    """Read credentials from Streamlit Cloud Secrets or environment."""

    try:
        consumer_key = st.secrets.get(
            "NEO_CONSUMER_KEY",
            "",
        )

        environment = st.secrets.get(
            "NEO_ENVIRONMENT",
            "prod",
        )

    except Exception:
        consumer_key = ""
        environment = "prod"

    if consumer_key:
        os.environ["NEO_CONSUMER_KEY"] = str(
            consumer_key
        ).strip()

    if environment:
        os.environ["NEO_ENVIRONMENT"] = str(
            environment
        ).strip()

    key = os.getenv(
        "NEO_CONSUMER_KEY",
        "",
    ).strip()

    return bool(
        key and not key.startswith("YOUR_")
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


def clear_stock_data():
    """Clear old market data and dependent session results."""

    st.session_state["price_data"] = None
    st.session_state["active_symbol"] = None
    st.session_state["data_source"] = None
    st.session_state["data_error"] = None
    st.session_state["fetch_signature"] = None
    st.session_state["scanner_results"] = None


# ============================================================
# CACHED KOTAK MARKET DATA
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
    """
    Automatically retrieve historical NSE daily candles.

    Successful data is cached for 30 minutes.
    """

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
# CACHED COMPANY NAME
# ============================================================

@st.cache_data(
    ttl=86400,
    show_spinner=False,
)
def get_company_profile(symbol):
    """
    Retrieve the full company name and basic profile.

    Yahoo Finance symbol format:
        RELIANCE -> RELIANCE.NS
    """

    import yfinance as yf

    ticker_symbol = (
        symbol
        if symbol.endswith(".NS")
        else f"{symbol}.NS"
    )

    ticker = yf.Ticker(ticker_symbol)

    info = ticker.get_info()

    if not isinstance(info, dict) or not info:
        raise ValueError(
            "Company profile is unavailable."
        )

    return {
        "company_name": (
            info.get("longName")
            or info.get("shortName")
            or symbol
        ),
        "sector": info.get("sector", "Not available"),
        "industry": info.get("industry", "Not available"),
    }


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
    """
    Retrieve annual financials and calculate:
    EPS, P/E, growth, ROE, ROCE, debt and valuation scenarios.
    """

    return fetch_auto_fundamentals(
        symbol=symbol,
        market_price=market_price,
        forecast_years=LONG_TERM_YEARS,
    )


# ============================================================
# AUTOMATIC AI FORECAST ENGINE
# ============================================================

@st.cache_data(
    ttl=21600,
    show_spinner=False,
)
def get_ai_forecasts(feature_df, horizons):
    """
    Automatically train eligible models.

    Forecast horizons are processed separately so one failure
    does not prevent the other horizons from being calculated.

    Returns forecast table, evaluation table and errors.
    """

    forecast_results = []
    evaluation_results = []
    errors = []

    for horizon in horizons:

        try:

            trained = train_forecast_models(
                feature_df=feature_df,
                horizons=[horizon],
            )

            forecast = predict_latest(
                feature_df=feature_df,
                trained_models=trained,
            )

            result = trained[horizon]

            forecast_results.append(forecast)

            evaluation_results.append({
                "horizon_days": horizon,
                "directional_accuracy_pct": (
                    result["directional_accuracy"] * 100
                ),
                "validation_rows": result["validation_rows"],
                "mae": result["mae"],
            })

        except Exception as exc:

            errors.append(
                f"{horizon}-day forecast: {exc}"
            )

    if forecast_results:

        forecasts = pd.concat(
            forecast_results,
            ignore_index=True,
        )

    else:

        forecasts = pd.DataFrame(
            columns=[
                "horizon_days",
                "current_price",
                "predicted_return_pct",
                "predicted_price",
            ]
        )

    evaluation = pd.DataFrame(
        evaluation_results
    )

    return forecasts, evaluation, errors


# ============================================================
# FORMATTING HELPERS
# ============================================================

def money(value):

    if value is None or pd.isna(value):
        return "N/A"

    return f"₹{float(value):,.2f}"


def percent(value):

    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):+.2f}%"


def ratio(value):

    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):.2f}x"


# ============================================================
# SHORT-TERM TECHNICAL LEVELS
# ============================================================

def calculate_short_term_levels(
    price_df,
    feature_df,
):
    """
    Calculate indicative technical levels.

    Entry zone:
        Between 20-day low support and SMA 20.

    Target 1:
        Entry + 1.5 ATR.

    Target 2:
        Entry + 3 ATR.

    Stop:
        Entry - 1.5 ATR.

    These are technical reference levels, not probability-based
    forecasts or guaranteed targets.
    """

    if len(feature_df) < 20:
        raise ValueError(
            "At least 20 daily records are required."
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

    recent = price_df.tail(20)

    support = float(
        pd.to_numeric(
            recent["low"],
            errors="coerce",
        ).min()
    )

    if not np.isfinite(support) or support <= 0:
        raise ValueError(
            "Recent support could not be calculated."
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
            "Technical stop level is invalid."
        )

    if current_price > entry_high:
        status = "Above preferred entry zone"

    elif current_price >= entry_low:
        status = "Inside preferred entry zone"

    else:
        status = "Below recent support area"

    return {
        "current_price": current_price,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "fair_entry": fair_entry,
        "target1": target1,
        "target2": target2,
        "stop_loss": stop_loss,
        "entry_status": status,
        "atr": atr_value,
    }


# ============================================================
# LONG-TERM ENTRY AND CAGR
# ============================================================

def calculate_long_term_levels(fundamental_data, current_price):
    """
    Calculate five-year Bear/Base/Bull targets.

    Expected CAGR:
        Calculated from current market price to Base target.

    Fair entry:
        Base target discounted using a 15% annual return hurdle.

    This is a target-based entry calculation, not a full DCF model.
    """

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
            "The Base valuation scenario is unavailable."
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
            "Invalid inputs for long-term valuation."
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
# 1. HOMEPAGE HEADER
# ============================================================

st.title("📈 StockVision AI")

st.caption(
    "Intelligent Indian Stock Analysis · Entry Prices · "
    "Short-Term and Long-Term Targets"
)


# ============================================================
# 2. SEARCH BAR AT THE TOP
# ============================================================

with st.form("stock_search_form"):

    search_col, button_col = st.columns(
        [6, 1]
    )

    with search_col:

        search_query = st.text_input(
            "Search NSE Company or Stock Symbol",
            value=st.session_state["selected_symbol"],
            placeholder="Example: RELIANCE, TCS, INFY, HDFCBANK",
        )

    with button_col:

        st.write("")
        st.write("")

        search_clicked = st.form_submit_button(
            "🔍 Analyse",
            use_container_width=True,
        )


if search_clicked:

    new_symbol = search_query.strip().upper()

    if not re.fullmatch(
        r"[A-Z0-9&.-]{1,30}",
        new_symbol,
    ):

        st.error(
            "Enter a valid NSE stock symbol."
        )

    else:

        if new_symbol != st.session_state["selected_symbol"]:

            st.session_state["selected_symbol"] = new_symbol

            clear_stock_data()

            st.rerun()

        else:

            st.session_state["fetch_signature"] = None

            st.rerun()


symbol = st.session_state["selected_symbol"]


# ============================================================
# 3. SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Settings")

period_label = st.sidebar.selectbox(
    "Historical Market Data",
    options=list(PERIOD_OPTIONS.keys()),
    index=0,
)

period_days = PERIOD_OPTIONS[period_label]

today = INDIA_TODAY

start_date = today - timedelta(
    days=period_days
)

end_date = today

refresh = st.sidebar.button(
    "↻ Refresh Stock Data",
    use_container_width=True,
)

with st.sidebar.expander("Upload Historical CSV"):

    csv_file = st.file_uploader(
        "OHLCV CSV",
        type=["csv"],
        key="market_csv",
    )

    if st.button("Load CSV"):

        if csv_file is None:

            st.warning(
                "Please upload a CSV file first."
            )

        else:

            try:

                uploaded = pd.read_csv(
                    csv_file
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
                    column
                    for column in required
                    if column not in uploaded.columns
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
                        "CSV loaded successfully."
                    )

            except Exception as exc:

                st.error(
                    f"CSV loading failed: {exc}"
                )


st.sidebar.divider()

st.sidebar.caption(
    "Historical price data is cached for 30 minutes."
)


# ============================================================
# 4. AUTOMATIC MARKET DATA LOADING
# ============================================================

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
                f"Loading {symbol} historical prices..."
            ):

                prices = get_market_data(
                    symbol=symbol,
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    credential_hash=CREDENTIAL_HASH,
                )

            st.session_state["price_data"] = prices

            st.session_state["active_symbol"] = symbol

            st.session_state["data_source"] = "Kotak Neo"

        except Exception as exc:

            st.session_state["data_error"] = str(exc)

    else:

        st.session_state["data_error"] = (
            "Kotak API credentials are missing. Configure "
            "NEO_CONSUMER_KEY in Streamlit Secrets."
        )


# ============================================================
# 5. CHECK MARKET DATA
# ============================================================

price_df = st.session_state["price_data"]

active_symbol = (
    st.session_state["active_symbol"]
    or symbol
)

if price_df is None:

    st.warning(
        st.session_state["data_error"]
        or "Historical market data is not available."
    )

    st.info(
        "Choose an NSE symbol and configure Kotak Neo credentials. "
        "StockVision AI will automatically download historical prices."
    )

    st.stop()


# ============================================================
# 6. TECHNICAL FEATURE CALCULATION
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
        "No usable historical market observations."
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

daily_change_pct = (
    (current_price / previous_price - 1) * 100
    if previous_price > 0
    else 0
)

latest_date = pd.to_datetime(
    latest["date"]
).date()


# ============================================================
# 7. COMPANY PROFILE
# ============================================================

try:

    profile = get_company_profile(
        active_symbol
    )

    company_name = profile["company_name"]

    sector = profile["sector"]

    industry = profile["industry"]

except Exception:

    company_name = active_symbol

    sector = "Not available"

    industry = "Not available"


# ============================================================
# 8. COMPANY NAME AND MARKET SUMMARY
# ============================================================

st.header(company_name)

st.caption(
    f"NSE: {active_symbol} · "
    f"{sector} · "
    f"{industry}"
)

st.caption(
    f"Market data: {st.session_state['data_source']} · "
    f"Latest available candle: {latest_date:%d %b %Y}"
)

a, b, c, d = st.columns(4)

a.metric(
    "Current Price",
    money(current_price),
    percent(daily_change_pct),
)

rsi = latest.get(
    "rsi_14",
    np.nan,
)

b.metric(
    "RSI (14)",
    f"{rsi:.2f}" if pd.notna(rsi) else "N/A",
)

c.metric(
    "Daily Volume",
    f"{float(latest['volume']):,.0f}",
)

d.metric(
    "Trading Records",
    f"{len(feature_df):,}",
)

st.divider()


# ============================================================
# 9. AUTOMATIC SHORT-TERM PRICE LEVELS
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
# 10. AUTOMATIC FUNDAMENTALS
# ============================================================

fundamentals = None

long_levels = None

fundamental_error = None

try:

    with st.spinner(
        f"Calculating {active_symbol} long-term valuation..."
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
# 11. MAIN PRICE LEVEL DASHBOARD
# ============================================================

st.header("🎯 Fair Entry Prices & Targets")

st.caption(
    "Automatically calculated from current technical indicators "
    "and retrieved company financial statements."
)


# ============================================================
# SHORT-TERM PRICE CARD
# ============================================================

short_col, long_col = st.columns(2)

with short_col:

    st.subheader("⚡ Short-Term Outlook")

    st.caption(
        "Technical levels · Approx. 21 trading days"
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

        x, y = st.columns(2)

        x.metric(
            "Target 1",
            money(short_levels["target1"]),
        )

        y.metric(
            "Target 2",
            money(short_levels["target2"]),
        )

        st.metric(
            "Risk / Stop-Loss Level",
            money(short_levels["stop_loss"]),
        )

        st.caption(
            "Entry is based on 20-day support and SMA 20. "
            "Targets and stop-loss use ATR multiples."
        )

    else:

        st.warning(
            f"Short-term levels unavailable: {short_error}"
        )


# ============================================================
# LONG-TERM PRICE CARD
# ============================================================

with long_col:

    st.subheader("🏦 Long-Term Outlook")

    st.caption(
        "Five-year fundamental valuation"
    )

    if long_levels is not None:

        st.metric(
            "Fair Entry Price",
            money(long_levels["fair_entry"]),
            "15% annual return hurdle",
        )

        st.metric(
            "Expected Base-Case CAGR",
            percent(long_levels["expected_cagr_pct"]),
            "Based on current market price",
        )

        x, y, z = st.columns(3)

        x.metric(
            "Bear Target",
            money(long_levels["bear_target"]),
        )

        y.metric(
            "Base Target",
            money(long_levels["base_target"]),
        )

        z.metric(
            "Bull Target",
            money(long_levels["bull_target"]),
        )

        current_vs_entry = (
            (current_price / long_levels["fair_entry"]) - 1
        ) * 100

        st.caption(
            f"Current price is {current_vs_entry:+.2f}% "
            "relative to the calculated fair-entry level."
        )

    else:

        st.warning(
            "Long-term valuation unavailable. "
            "Verified financial data is needed."
        )

        st.caption(
            fundamental_error
        )


# ============================================================
# 12. PRICE COMPARISON CHART
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
                "Level": "Short-Term Entry",
                "Price": short_levels["fair_entry"],
                "Category": "Short Term",
            },
            {
                "Level": "Short Target 1",
                "Price": short_levels["target1"],
                "Category": "Short Term",
            },
            {
                "Level": "Short Target 2",
                "Price": short_levels["target2"],
                "Category": "Short Term",
            },
        ])

    if long_levels is not None:

        price_levels.extend([
            {
                "Level": "Long-Term Entry",
                "Price": long_levels["fair_entry"],
                "Category": "Long Term",
            },
            {
                "Level": "Five-Year Base Target",
                "Price": long_levels["base_target"],
                "Category": "Long Term",
            },
        ])

    comparison_df = pd.DataFrame(
        price_levels
    )

    price_chart = px.bar(
        comparison_df,
        x="Level",
        y="Price",
        color="Category",
        title=f"{active_symbol} Price Levels",
    )

    price_chart.update_layout(
        height=350,
        yaxis_title="Price (₹)",
        xaxis_title="",
        margin=dict(
            l=5,
            r=5,
            t=45,
            b=5,
        ),
    )

    st.plotly_chart(
        price_chart,
        use_container_width=True,
    )


# ============================================================
# 13. AUTOMATIC AI FORECASTING
# ============================================================

with st.expander(
    "🤖 AI Forecast Outlook",
    expanded=False,
):

    st.caption(
        "Eligible models are trained automatically. "
        "No manual training controls are required."
    )

    # Approximate minimum dataset sizes for this project's
    # current 200-day moving-average and validation requirements.
    eligible_horizons = [
        horizon
        for horizon, minimum_rows in [
            (21, 450),
            (63, 550),
            (126, 700),
            (252, 1050),
        ]
        if len(feature_df) >= minimum_rows
    ]

    if not eligible_horizons:

        st.info(
            f"Currently loaded: {len(feature_df)} records. "
            "More history is needed for reliable model training. "
            "Choose a longer historical range in the sidebar."
        )

    else:

        try:

            with st.spinner(
                "Generating AI price estimates..."
            ):

                forecasts, evaluation, forecast_errors = (
                    get_ai_forecasts(
                        feature_df,
                        tuple(eligible_horizons),
                    )
                )

            if not forecasts.empty:

                st.subheader(
                    "AI-Estimated Future Prices"
                )

                forecast_columns = st.columns(
                    min(len(forecasts), 4)
                )

                for column, (_, row) in zip(
                    forecast_columns,
                    forecasts.iterrows(),
                ):

                    with column:

                        st.metric(
                            f"{int(row['horizon_days'])}-Day Estimate",
                            money(row["predicted_price"]),
                            percent(row["predicted_return_pct"]),
                        )

                st.subheader(
                    "Historical Model Evaluation"
                )

                st.dataframe(
                    evaluation.rename(
                        columns={
                            "horizon_days": "Horizon (Days)",
                            "directional_accuracy_pct": "Directional Accuracy (%)",
                            "validation_rows": "Validation Records",
                            "mae": "Mean Absolute Error",
                        }
                    ).style.format({
                        "Directional Accuracy (%)": "{:.2f}%",
                        "Mean Absolute Error": "{:.5f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "Historical directional accuracy is not the "
                    "probability of a profitable future trade."
                )

            if forecast_errors:

                with st.expander("Forecast availability details"):

                    for error in forecast_errors:
                        st.caption(error)

        except Exception as exc:

            st.info(
                f"AI forecast calculations are unavailable: {exc}"
            )


# ============================================================
# 14. COMPANY FUNDAMENTALS
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
            "Annual company financial information"
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "EPS",
            money(metrics["eps"]),
        )

        b.metric(
            "P/E",
            ratio(metrics["pe_ratio"]),
        )

        c.metric(
            "Revenue CAGR",
            percent(metrics["revenue_growth_pct"]),
        )

        d.metric(
            "EPS CAGR",
            percent(metrics["eps_growth_pct"]),
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "ROE",
            percent(metrics["roe_pct"]),
        )

        b.metric(
            "ROCE",
            percent(metrics["roce_pct"]),
        )

        c.metric(
            "Operating Margin",
            percent(metrics["operating_margin_pct"]),
        )

        d.metric(
            "Net Profit Margin",
            percent(metrics["net_profit_margin_pct"]),
        )

        a, b = st.columns(2)

        a.metric(
            "Debt / Equity",
            ratio(metrics["debt_to_equity"]),
        )

        b.metric(
            "Current Ratio",
            ratio(metrics["current_ratio"]),
        )

        with st.expander("Annual Financial History"):

            history = fundamentals["financial_history"]

            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Valuation Methodology"):

            st.write(
                fundamentals["assumptions"]
            )

            st.write(
                f"EPS source: {metrics['eps_source']}"
            )

            st.write(
                f"Financial data source: {fundamentals['source']}"
            )

    else:

        st.info(
            "Financial statements are not currently available "
            "for this company. No financial values have been invented."
        )

        if fundamental_error:
            st.caption(fundamental_error)


# ============================================================
# 15. TECHNICAL ANALYSIS
# ============================================================

with st.expander(
    "📊 Detailed Technical Analysis",
    expanded=False,
):

    chart_columns = [
        column
        for column in [
            "close",
            "sma_20",
            "sma_50",
            "sma_200",
        ]
        if column == "close"
        or (
            column in feature_df.columns
            and feature_df[column].notna().any()
        )
    ]

    chart = px.line(
        feature_df,
        x="date",
        y=chart_columns,
        title=f"{active_symbol} Price and Moving Averages",
    )

    chart.update_layout(
        height=400,
        hovermode="x unified",
        yaxis_title="Price (₹)",
        legend_title_text="",
    )

    st.plotly_chart(
        chart,
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
            "Score",
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
# 16. AUTOMATIC RISK ANALYSIS
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
            percent(risk["annualized_volatility_pct"]),
        )

        b.metric(
            "Maximum Drawdown",
            percent(risk["maximum_drawdown_pct"]),
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
            percent(risk["daily_var_pct"]),
        )

        a, b, c = st.columns(3)

        a.metric(
            "Sortino Ratio",
            (
                f"{risk['sortino_ratio']:.2f}"
                if pd.notna(risk["sortino_ratio"])
                else "N/A"
            ),
        )

        b.metric(
            "Conditional VaR",
            percent(risk["daily_cvar_pct"]),
        )

        c.metric(
            "Historical Return",
            percent(risk["annualized_return_pct"]),
        )

        st.caption(
            "Backward-looking risk measurements are not guarantees "
            "of future gains or maximum losses."
        )

    except Exception as exc:

        st.warning(
            f"Risk calculations unavailable: {exc}"
        )


# ============================================================
# 17. WATCHLIST
# ============================================================

with st.expander(
    "⭐ Watchlist",
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
            "Add Stock",
            use_container_width=True,
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
            "No stocks have been added yet."
        )

    else:

        st.dataframe(
            watchlist,
            use_container_width=True,
            hide_index=True,
        )

        remove_choice = st.selectbox(
            "Remove stock",
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

        st.divider()

        if st.button(
            "Scan Watchlist",
            type="primary",
        ):

            if not HAS_KOTAK_KEY:

                st.warning(
                    "Kotak credentials are required for the scanner."
                )

            else:

                try:

                    with st.spinner(
                        "Analysing watchlist..."
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
                        f"Watchlist scan failed: {exc}"
                    )

        results = st.session_state["scanner_results"]

        if results is not None:

            st.subheader("Watchlist Scan Results")

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Scanner CSV",
                data=results.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="stockvision_watchlist.csv",
                mime="text/csv",
            )


# ============================================================
# 18. TRANSPARENCY AND DISCLAIMER
# ============================================================

with st.expander(
    "How StockVision calculates entry prices and targets",
):

    st.markdown(
        """
        **Short-term entry:** Midpoint between 20-day price support
        and the 20-day simple moving average.

        **Short-term targets:** ATR multiples above the entry reference.
        These are technical scenario levels, not guaranteed forecasts.

        **Long-term targets:** Five-year EPS growth and future P/E
        scenarios using company financial statements.

        **Long-term fair entry:** Five-year Base target discounted
        using a 15% annual required return.

        **Expected CAGR:** Annualized return implied by the five-year
        Base target relative to today's market price.
        """
    )

    st.caption(
        "Yahoo Finance data is unofficial and may be delayed, "
        "incomplete or unavailable. Short-term technical levels "
        "and long-term valuation scenarios are estimates, not "
        "guaranteed price targets or personalized investment advice."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "StockVision AI · Research and analysis platform · "
    "All entry prices, target prices and expected returns are estimates."
)
