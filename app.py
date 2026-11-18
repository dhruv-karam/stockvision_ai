
import os
import hashlib
from datetime import date, timedelta

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

from stockvision.modeling import (
    FORECAST_HORIZONS,
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

from stockvision.backtesting import (
    walk_forward_backtest,
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
    initial_sidebar_state="expanded",
)

LONG_TERM_YEARS = 5
REQUIRED_RETURN = 0.15

PERIOD_OPTIONS = {
    "1 Year": 365,
    "3 Years": 1095,
    "5 Years": 1825,
}


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(125,135,150,0.07);
        border: 1px solid rgba(150,150,150,0.16);
        border-radius: 12px;
        padding: 13px;
    }

    h1 {
        letter-spacing: -0.04em;
    }

    div[data-testid="stExpander"] {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SECRETS
# ============================================================

def load_credentials():

    try:
        key = st.secrets.get("NEO_CONSUMER_KEY", "")
        environment = st.secrets.get(
            "NEO_ENVIRONMENT", "prod"
        )
    except Exception:
        key = ""
        environment = "prod"

    if key:
        os.environ["NEO_CONSUMER_KEY"] = str(key).strip()

    if environment:
        os.environ["NEO_ENVIRONMENT"] = str(environment).strip()

    return bool(key and str(key).strip())


HAS_KOTAK_KEY = load_credentials()

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
    "price_data": None,
    "active_symbol": None,
    "data_source": None,
    "data_error": None,
    "fetch_signature": None,
    "trained_models": None,
    "forecasts": None,
    "backtest_summary": None,
    "backtest_details": None,
    "scanner_results": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clear_old_analysis():

    for key in (
        "trained_models",
        "forecasts",
        "backtest_summary",
        "backtest_details",
        "scanner_results",
    ):
        st.session_state[key] = None


# ============================================================
# CACHED MARKET DATA
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
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
            item["error"] for item in lookup["failed"]
        )

        raise RuntimeError(
            f"Instrument lookup failed: {errors}"
        )

    prices = fetch_daily_history(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        client=client,
    )

    if prices.empty:
        raise RuntimeError(
            f"No historical data returned for {symbol}."
        )

    return prices


# ============================================================
# CACHED FUNDAMENTALS
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def get_company_fundamentals(
    symbol,
    market_price,
):

    return fetch_auto_fundamentals(
        symbol=symbol,
        market_price=market_price,
        forecast_years=LONG_TERM_YEARS,
    )


# ============================================================
# PRICE LEVEL CALCULATOR — SHORT TERM
# ============================================================

def calculate_short_term_levels(price_data, feature_data):
    """
    Technical entry and target estimates for 21 trading days.

    Entry:
        Midpoint of recent 20-day support and SMA 20.

    Target 1:
        Entry + 0.5 x ATR x sqrt(21)

    Target 2:
        Entry + 1.0 x ATR x sqrt(21)

    Stop:
        Entry - 0.5 x ATR x sqrt(21)

    These are volatility-based scenario levels, not
    statistically calibrated price forecasts.
    """

    if len(price_data) < 20 or len(feature_data) < 20:
        raise ValueError(
            "At least 20 daily observations are required."
        )

    latest = feature_data.iloc[-1]

    sma20 = latest.get("sma_20")
    atr = latest.get("atr_14")

    if pd.isna(sma20) or pd.isna(atr):
        raise ValueError(
            "Insufficient observations for SMA 20 or ATR."
        )

    if float(sma20) <= 0 or float(atr) <= 0:
        raise ValueError(
            "Invalid SMA or ATR values."
        )

    recent = price_data.tail(20)

    support = float(recent["low"].min())
    current_price = float(latest["close"])

    entry_low = min(support, float(sma20))
    entry_high = max(support, float(sma20))

    fair_entry = (entry_low + entry_high) / 2

    expected_move = float(atr) * np.sqrt(21)

    target1 = fair_entry + 0.5 * expected_move
    target2 = fair_entry + expected_move

    stop_loss = fair_entry - 0.5 * expected_move

    if fair_entry <= 0 or stop_loss <= 0:
        raise ValueError(
            "Calculated price levels are outside the valid range."
        )

    if current_price > entry_high:
        entry_status = "Above preferred entry zone"
    elif current_price >= entry_low:
        entry_status = "Inside entry zone"
    else:
        entry_status = "Below support zone"

    return {
        "current_price": current_price,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "fair_entry": fair_entry,
        "target1": target1,
        "target2": target2,
        "stop_loss": stop_loss,
        "expected_move": expected_move,
        "entry_status": entry_status,
    }


# ============================================================
# PRICE LEVEL CALCULATOR — LONG TERM
# ============================================================

def calculate_long_term_levels(fundamentals):
    """
    Discount the five-year Base scenario target using a
    15% annual required return.

    Long-term fair entry:
        Base future price / (1 + required return) ** years
    """

    scenarios = fundamentals["scenarios"]

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
            "Base-case valuation is unavailable."
        )

    base_target = float(
        base.iloc[0]["estimated_future_price"]
    )

    fair_entry = (
        base_target
        / ((1 + REQUIRED_RETURN) ** LONG_TERM_YEARS)
    )

    if not np.isfinite(fair_entry) or fair_entry <= 0:
        raise ValueError(
            "Invalid long-term fair-entry calculation."
        )

    return {
        "fair_entry": fair_entry,
        "bear_target": (
            float(bear.iloc[0]["estimated_future_price"])
            if not bear.empty else np.nan
        ),
        "base_target": base_target,
        "bull_target": (
            float(bull.iloc[0]["estimated_future_price"])
            if not bull.empty else np.nan
        ),
        "years": LONG_TERM_YEARS,
        "required_return": REQUIRED_RETURN,
    }


# ============================================================
# FORMATTING
# ============================================================

def money(value):

    if value is None or pd.isna(value):
        return "N/A"

    return f"₹{float(value):,.2f}"


def percent(value):

    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):+.2f}%"


# ============================================================
# HEADER
# ============================================================

st.title("📈 StockVision AI")

st.caption(
    "Smart stock analysis · Indicative entry prices · "
    "Short-term and long-term targets"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Stock Selection")

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = "RELIANCE"

symbol = st.sidebar.text_input(
    "NSE Symbol",
    key="selected_symbol",
).strip().upper()

period_label = st.sidebar.selectbox(
    "Historical data",
    list(PERIOD_OPTIONS.keys()),
    index=0,
)

period_days = PERIOD_OPTIONS[period_label]

today = date.today()

start_date = today - timedelta(days=period_days)
end_date = today

refresh = st.sidebar.button(
    "↻ Refresh Data",
    use_container_width=True,
)

with st.sidebar.expander("Upload CSV instead"):

    csv_file = st.file_uploader(
        "OHLCV data",
        type=["csv"],
        key="fallback_csv",
    )

    if st.button("Load CSV"):

        if csv_file is None:
            st.warning("Upload a CSV first.")
        else:
            try:
                uploaded = pd.read_csv(csv_file)

                required = [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]

                missing = [
                    c for c in required if c not in uploaded.columns
                ]

                if missing:
                    st.error(
                        "Missing columns: " + ", ".join(missing)
                    )
                else:
                    st.session_state.price_data = uploaded[required].copy()
                    st.session_state.active_symbol = symbol
                    st.session_state.data_source = "CSV upload"
                    st.session_state.data_error = None
                    clear_old_analysis()

                    st.rerun()

            except Exception as exc:
                st.error(f"CSV loading error: {exc}")


request_signature = (
    f"{symbol}|{period_days}|{CREDENTIAL_HASH}"
)

if refresh:
    get_market_data.clear()
    st.session_state.fetch_signature = None


# ============================================================
# AUTOMATIC DATA LOADING
# ============================================================

if (
    symbol
    and st.session_state.fetch_signature != request_signature
    and st.session_state.data_source != "CSV upload"
):

    st.session_state.fetch_signature = request_signature
    st.session_state.price_data = None
    st.session_state.active_symbol = None
    st.session_state.data_source = None
    st.session_state.data_error = None

    clear_old_analysis()

    if HAS_KOTAK_KEY:

        try:

            with st.spinner(
                f"Loading {symbol} market data..."
            ):

                prices = get_market_data(
                    symbol,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    CREDENTIAL_HASH,
                )

            st.session_state.price_data = prices
            st.session_state.active_symbol = symbol
            st.session_state.data_source = "Kotak Neo"

        except Exception as exc:
            st.session_state.data_error = str(exc)

    else:

        st.session_state.data_error = (
            "Kotak credentials are missing. Configure NEO_CONSUMER_KEY "
            "in Streamlit Secrets or upload an OHLCV CSV."
        )


# ============================================================
# CHECK DATA
# ============================================================

price_df = st.session_state.price_data

active_symbol = (
    st.session_state.active_symbol or symbol
)

if price_df is None:

    st.warning(
        st.session_state.data_error
        or "No market data is loaded."
    )

    st.info(
        "Select a stock from the sidebar. StockVision AI will "
        "automatically fetch data when Kotak access is available."
    )

    st.stop()


# ============================================================
# CALCULATE INDICATORS
# ============================================================

try:
    feature_df = build_features(price_df)
except Exception as exc:
    st.error(f"Indicator calculation failed: {exc}")
    st.stop()

if feature_df.empty:
    st.error("No usable price observations.")
    st.stop()

latest = feature_df.iloc[-1]

current_price = float(latest["close"])

latest_date = pd.to_datetime(latest["date"]).date()

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


# ============================================================
# MAIN HOME SUMMARY
# ============================================================

st.subheader(
    f"{active_symbol} · Stock Analysis"
)

st.caption(
    f"{st.session_state.data_source} · "
    f"{latest_date:%d %b %Y} · "
    f"{len(feature_df):,} daily records"
)

a, b, c, d = st.columns(4)

a.metric(
    "Current Market Price",
    money(current_price),
    percent(daily_change),
)

a_placeholder = latest.get("rsi_14")

b.metric(
    "RSI (14)",
    f"{a_placeholder:.2f}"
    if pd.notna(a_placeholder)
    else "N/A",
)

c.metric(
    "Daily Volume",
    f"{float(latest['volume']):,.0f}",
)

d.metric(
    "Data Records",
    f"{len(feature_df):,}",
)

st.divider()


# ============================================================
# SHORT TERM AND LONG TERM ENTRY LEVELS
# ============================================================

st.header("🎯 Entry Prices & Target Levels")

st.caption(
    "Automatically calculated from technical price history "
    "and annual company financial statements."
)

short_levels = None
long_levels = None
fundamentals = None

short_error = None
long_error = None


# ============================================================
# SHORT-TERM LEVELS
# ============================================================

try:

    short_levels = calculate_short_term_levels(
        price_df,
        feature_df,
    )

except Exception as exc:
    short_error = str(exc)


# ============================================================
# LONG-TERM FUNDAMENTALS
# ============================================================

try:

    with st.spinner(
        f"Calculating {active_symbol} long-term valuation..."
    ):

        fundamentals = get_company_fundamentals(
            active_symbol,
            current_price,
        )

    long_levels = calculate_long_term_levels(
        fundamentals
    )

except Exception as exc:
    long_error = str(exc)


# ============================================================
# DISPLAY BOTH HORIZONS
# ============================================================

short_col, long_col = st.columns(2)

# ------------------------------------------------------------
# SHORT TERM CARD
# ------------------------------------------------------------

with short_col:

    st.subheader("⚡ Short Term")

    st.caption("21-trading-day technical scenario")

    if short_levels is not None:

        st.metric(
            "Indicative Fair Entry",
            money(short_levels["fair_entry"]),
            short_levels["entry_status"],
        )

        st.write(
            "**Preferred entry zone:** "
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
            "Risk / Stop Level",
            money(short_levels["stop_loss"]),
        )

        st.caption(
            "Entry uses the midpoint between 20-day support "
            "and SMA 20. Targets and stop are ATR-based volatility "
            "scenarios, not model predictions."
        )

    else:

        st.warning(
            "Short-term levels unavailable: " + str(short_error)
        )


# ------------------------------------------------------------
# LONG TERM CARD
# ------------------------------------------------------------

with long_col:

    st.subheader("🏦 Long Term")

    st.caption("Five-year fundamental valuation scenario")

    if long_levels is not None:

        current_fair_entry = long_levels["fair_entry"]

        gap_pct = (
            (current_price / current_fair_entry) - 1
        ) * 100

        st.metric(
            "Indicative Fair Entry",
            money(current_fair_entry),
            f"{gap_pct:+.2f}% vs current price",
        )

        st.metric(
            "Bear Target",
            money(long_levels["bear_target"]),
        )

        st.metric(
            "Base Target",
            money(long_levels["base_target"]),
        )

        st.metric(
            "Bull Target",
            money(long_levels["bull_target"]),
        )

        st.caption(
            "Fair entry discounts the five-year Base target "
            f"at a {long_levels['required_return']:.0%} annual "
            "required return."
        )

    else:

        st.warning(
            "Long-term valuation unavailable: "
            + str(long_error)
        )


# ============================================================
# PRICE LEVEL COMPARISON
# ============================================================

if short_levels is not None or long_levels is not None:

    st.divider()

    st.subheader("Current Price vs Entry Estimates")

    comparison = [
        {
            "Price Level": "Current Price",
            "Price": current_price,
            "Type": "Market",
        }
    ]

    if short_levels:

        comparison.extend([
            {
                "Price Level": "Short-term Entry",
                "Price": short_levels["fair_entry"],
                "Type": "Short Term",
            },
            {
                "Price Level": "Short Target 1",
                "Price": short_levels["target1"],
                "Type": "Short Term",
            },
            {
                "Price Level": "Short Target 2",
                "Price": short_levels["target2"],
                "Type": "Short Term",
            },
        ])

    if long_levels:

        comparison.extend([
            {
                "Price Level": "Long-term Entry",
                "Price": long_levels["fair_entry"],
                "Type": "Long Term",
            },
            {
                "Price Level": "Five-year Base Target",
                "Price": long_levels["base_target"],
                "Type": "Long Term",
            },
        ])

    comparison_df = pd.DataFrame(comparison)

    level_chart = px.bar(
        comparison_df,
        x="Price Level",
        y="Price",
        color="Type",
        title=f"{active_symbol} · Price Level Comparison",
    )

    level_chart.update_layout(
        height=350,
        margin=dict(l=5, r=5, t=40, b=5),
        yaxis_title="Price (₹)",
        xaxis_title="",
    )

    st.plotly_chart(
        level_chart,
        use_container_width=True,
    )


# ============================================================
# VALUATION AND DATA TRANSPARENCY
# ============================================================

with st.expander("How were these prices calculated?"):

    st.markdown(
        """
        **Short-term entry**

        Calculated from recent 20-day price support and the
        20-day simple moving average.

        **Short-term targets**

        Derived from Average True Range and the square-root-of-time
        volatility assumption over 21 trading days.

        **Long-term entry**

        The five-year Base-case estimated price is discounted
        using a 15% annual required return.

        **Long-term targets**

        Derived from historical company EPS growth, with fallback
        to historical profit or revenue growth when EPS history
        is unavailable. Future P/E assumptions are based on the
        current P/E with Bear/Base/Bull adjustments.
        """
    )

    st.caption(
        "These methodologies are simplified estimates, not a complete "
        "discounted cash flow valuation or a guarantee of future prices."
    )


# ============================================================
# COMPANY FUNDAMENTALS
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
            f"Financial year: {fundamentals['latest_fiscal_year']} · "
            f"Source: {fundamentals['source']}"
        )

        x1, x2, x3, x4 = st.columns(4)

        x1.metric(
            "EPS",
            money(metrics["eps"]),
        )

        x2.metric(
            "P/E",
            (
                f"{metrics['pe_ratio']:.2f}x"
                if pd.notna(metrics["pe_ratio"])
                else "N/A"
            ),
        )

        x3.metric(
            "ROE",
            percent(metrics["roe_pct"]),
        )

        x4.metric(
            "ROCE",
            percent(metrics["roce_pct"]),
        )

        x1, x2, x3, x4 = st.columns(4)

        x1.metric(
            "Revenue CAGR",
            percent(metrics["revenue_growth_pct"]),
        )

        x2.metric(
            "EPS CAGR",
            percent(metrics["eps_growth_pct"]),
        )

        x3.metric(
            "Operating Margin",
            percent(metrics["operating_margin_pct"]),
        )

        x4.metric(
            "Net Margin",
            percent(metrics["net_profit_margin_pct"]),
        )

        x1, x2 = st.columns(2)

        x1.metric(
            "Debt / Equity",
            (
                f"{metrics['debt_to_equity']:.2f}x"
                if pd.notna(metrics["debt_to_equity"])
                else "N/A"
            ),
        )

        x2.metric(
            "Current Ratio",
            (
                f"{metrics['current_ratio']:.2f}x"
                if pd.notna(metrics["current_ratio"])
                else "N/A"
            ),
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

    else:

        st.info(
            "Automatic financial data is currently unavailable. "
            "Long-term fair entry and targets cannot be calculated "
            "without verified financial information."
        )


# ============================================================
# MARKET CHART
# ============================================================

with st.expander(
    "📊 Price Chart & Technical Indicators",
    expanded=True,
):

    chart_columns = [
        "close",
        "sma_20",
        "sma_50",
        "sma_200",
    ]

    available = [
        column
        for column in chart_columns
        if column == "close"
        or feature_df[column].notna().any()
    ]

    chart = px.line(
        feature_df,
        x="date",
        y=available,
        title=f"{active_symbol} Historical Price Trend",
    )

    chart.update_layout(
        height=420,
        hovermode="x unified",
        yaxis_title="Price (₹)",
        legend_title_text="",
    )

    st.plotly_chart(
        chart,
        use_container_width=True,
    )

    x, y = st.columns(2)

    with x:

        rsi_chart = px.line(
            feature_df,
            x="date",
            y="rsi_14",
            title="RSI (14)",
        )

        rsi_chart.add_hline(y=70, line_dash="dash")
        rsi_chart.add_hline(y=30, line_dash="dash")

        st.plotly_chart(
            rsi_chart,
            use_container_width=True,
        )

    with y:

        macd_chart = px.line(
            feature_df,
            x="date",
            y=["macd", "macd_signal"],
            title="MACD",
        )

        st.plotly_chart(
            macd_chart,
            use_container_width=True,
        )


# ============================================================
# TECHNICAL SIGNAL
# ============================================================

with st.expander(
    "🎯 Detailed Technical Signal",
):

    try:

        signal = generate_technical_signal(
            feature_df
        )

        st.metric(
            "Overall Technical Bias",
            signal["overall_signal"],
        )

        st.metric(
            "Technical Score",
            f"{signal['score']:+.1f} / 100",
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
# RISK ANALYSIS
# ============================================================

with st.expander(
    "⚠️ Historical Risk Analysis",
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
            format(risk["sharpe_ratio"], ".2f")
            if pd.notna(risk["sharpe_ratio"])
            else "N/A",
        )

        d.metric(
            "Daily VaR",
            percent(risk["daily_var_pct"]),
        )

        st.caption(
            "Historical risk estimates do not guarantee "
            "future performance or losses."
        )

    except Exception as exc:

        st.warning(
            f"Risk metrics unavailable: {exc}"
        )


# ============================================================
# AI FORECASTING
# ============================================================

with st.expander(
    "🤖 AI Forecasting Models",
):

    st.caption(
        "Train XGBoost models for forward-return estimation."
    )

    horizons = st.multiselect(
        "Forecast horizons",
        FORECAST_HORIZONS,
        default=[21, 63],
        format_func=lambda x: f"{x} trading days",
        key="model_horizons",
    )

    if st.button(
        "Train Forecast Models",
        type="primary",
    ):

        if not horizons:

            st.warning(
                "Select at least one forecast horizon."
            )

        else:

            try:

                with st.spinner(
                    "Training selected models..."
                ):

                    models = train_forecast_models(
                        feature_df,
                        horizons=horizons,
                    )

                    predictions = predict_latest(
                        feature_df,
                        models,
                    )

                st.session_state["trained_models"] = models
                st.session_state["forecasts"] = predictions

            except Exception as exc:

                st.error(
                    f"Forecast training failed: {exc}"
                )

    models = st.session_state["trained_models"]
    predictions = st.session_state["forecasts"]

    if models is not None and predictions is not None:

        st.subheader("Model Forecasts")

        st.dataframe(
            predictions,
            use_container_width=True,
            hide_index=True,
        )

        evaluation = pd.DataFrame([
            {
                "Horizon": f"{horizon} days",
                "MAE": result["mae"],
                "RMSE": result["rmse"],
                "Directional Accuracy": (
                    result["directional_accuracy"] * 100
                ),
            }
            for horizon, result in models.items()
        ])

        st.subheader("Model Evaluation")

        st.dataframe(
            evaluation,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# WALK-FORWARD BACKTESTING
# ============================================================

with st.expander(
    "🧪 Walk-Forward Backtesting",
):

    backtest_horizons = st.multiselect(
        "Backtest horizons",
        FORECAST_HORIZONS,
        default=[21, 63],
        format_func=lambda x: f"{x} days",
        key="backtest_horizons",
    )

    max_folds = st.slider(
        "Maximum test periods",
        3,
        20,
        8,
    )

    if st.button("Run Backtest"):

        if not backtest_horizons:

            st.warning(
                "Select at least one horizon."
            )

        else:

            try:

                with st.spinner(
                    "Running walk-forward backtesting..."
                ):

                    summary, details = walk_forward_backtest(
                        feature_df,
                        horizons=backtest_horizons,
                        min_train_rows=200,
                        max_folds=max_folds,
                    )

                st.session_state["backtest_summary"] = summary
                st.session_state["backtest_details"] = details

            except Exception as exc:

                st.error(
                    f"Backtesting failed: {exc}"
                )

    if st.session_state["backtest_summary"] is not None:

        st.subheader("Backtesting Performance")

        st.dataframe(
            st.session_state["backtest_summary"],
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# WATCHLIST
# ============================================================

with st.expander(
    "⭐ Watchlist & Multi-Stock Scanner",
):

    with st.form("watchlist_form", clear_on_submit=True):

        new_symbol = st.text_input(
            "Add NSE stock",
            placeholder="Example: TCS",
        )

        add_clicked = st.form_submit_button(
            "Add to Watchlist"
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
            "Select stock to remove",
            [""] + watchlist["symbol"].tolist(),
        )

        if st.button("Remove Stock"):

            if remove_choice:

                try:
                    remove_symbol(remove_choice)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        st.divider()

        if st.button(
            "Scan All Watchlist Stocks",
            type="primary",
        ):

            if not HAS_KOTAK_KEY:

                st.warning(
                    "Kotak credentials are required for scanning."
                )

            else:

                try:

                    with st.spinner(
                        "Scanning your NSE watchlist..."
                    ):

                        client = create_neo_client()

                        update_instruments(
                            watchlist["symbol"].tolist(),
                            client=client,
                        )

                        results = scan_watchlist(
                            watchlist_df=watchlist,
                            lookback_days=750,
                            client=client,
                        )

                    st.session_state["scanner_results"] = results

                except Exception as exc:

                    st.error(
                        f"Stock scanner failed: {exc}"
                    )

        scanner_results = st.session_state["scanner_results"]

        if scanner_results is not None:

            st.subheader("Scanner Results")

            st.dataframe(
                scanner_results,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Scanner CSV",
                data=scanner_results.to_csv(index=False).encode("utf-8"),
                file_name="stockvision_scanner.csv",
                mime="text/csv",
            )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "StockVision AI · Indicative research estimates only. "
    "Short-term levels are ATR-based scenarios. Long-term "
    "valuation depends on retrieved company financials and "
    "stated assumptions. Prices are not guaranteed targets "
    "or personalised investment advice."
)
