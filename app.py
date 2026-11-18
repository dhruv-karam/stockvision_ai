
import hashlib
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import streamlit as st

from stockvision.data import create_neo_client, fetch_daily_history
from stockvision.features import build_features
from stockvision.instruments import update_instruments
from stockvision.modeling import (
    FORECAST_HORIZONS,
    train_forecast_models,
    predict_latest,
)
from stockvision.valuation import calculate_long_term_scenarios
from stockvision.signals import generate_technical_signal
from stockvision.risk import calculate_risk_metrics
from stockvision.backtesting import walk_forward_backtest
from stockvision.watchlist import load_watchlist, add_symbol, remove_symbol
from stockvision.scanner import scan_watchlist
from stockvision.fundamentals import analyze_fundamentals


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="StockVision AI",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1440px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(125, 135, 150, 0.08);
        border: 1px solid rgba(150, 150, 150, 0.18);
        padding: 12px 15px;
        border-radius: 12px;
    }

    div[data-testid="stTabs"] button {
        padding: .6rem .9rem;
        font-weight: 600;
    }

    h1 {
        letter-spacing: -.03em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# STREAMLIT CLOUD SECRETS
# --------------------------------------------------

def apply_streamlit_secrets():
    """Make Streamlit Cloud secrets available to the Kotak client."""

    try:
        key = st.secrets.get("NEO_CONSUMER_KEY", "")
        environment = st.secrets.get("NEO_ENVIRONMENT", "prod")
    except Exception:
        key, environment = "", "prod"

    if key:
        os.environ["NEO_CONSUMER_KEY"] = str(key).strip()

    if environment:
        os.environ["NEO_ENVIRONMENT"] = str(environment).strip()


apply_streamlit_secrets()

consumer_key = os.getenv("NEO_CONSUMER_KEY", "").strip()

has_kotak_key = bool(
    consumer_key and not consumer_key.startswith("YOUR_")
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

DEFAULTS = {
    "price_data": None,
    "active_symbol": None,
    "data_source": None,
    "data_error": None,
    "auto_fetch_signature": None,
    "trained_models": None,
    "forecasts": None,
    "backtest_summary": None,
    "backtest_predictions": None,
    "valuation_results": None,
    "risk_results": None,
    "fundamental_results": None,
    "fundamental_symbol": None,
    "scanner_results": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def clear_analysis_results():
    """Clear calculations tied to an older price dataset."""

    for key in (
        "trained_models",
        "forecasts",
        "backtest_summary",
        "backtest_predictions",
        "valuation_results",
        "risk_results",
        "fundamental_results",
        "fundamental_symbol",
        "scanner_results",
    ):
        st.session_state[key] = None


# --------------------------------------------------
# AUTOMATIC KOTAK DATA FETCH
# --------------------------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_market_history_cached(
    symbol,
    start_text,
    end_text,
    credential_fingerprint,
):
    """
    Find the NSE instrument and fetch daily historical candles.

    Cached for 30 minutes to reduce unnecessary API requests.
    """

    # Included in Streamlit's cache key.
    _ = credential_fingerprint

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
            f"Instrument lookup failed for {symbol}. {errors}"
        )

    prices = fetch_daily_history(
        symbol=symbol,
        start_date=start_text,
        end_date=end_text,
        client=client,
    )

    if prices.empty:
        raise RuntimeError(
            f"Kotak Neo returned no daily candles for {symbol}."
        )

    return prices


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

today = datetime.now(
    ZoneInfo("Asia/Kolkata")
).date()

st.sidebar.title("StockVision AI")
st.sidebar.caption("Market data and analysis")

if "selected_symbol" not in st.session_state:
    st.session_state["selected_symbol"] = "RELIANCE"

symbol = st.sidebar.text_input(
    "NSE stock symbol",
    key="selected_symbol",
    help="Examples: RELIANCE, TCS, INFY, HDFCBANK",
).strip().upper()

periods = {
    "1 year": 365,
    "3 years": 1095,
    "5 years": 1825,
}

period_label = st.sidebar.selectbox(
    "Historical range",
    list(periods),
    index=0,
)

period_days = periods[period_label]

start_date = today - timedelta(days=period_days)
end_date = today

refresh = st.sidebar.button(
    "↻ Refresh market data",
    use_container_width=True,
)

request_signature = (
    symbol,
    period_days,
    hashlib.sha256(
        consumer_key.encode()
    ).hexdigest()[:16] if consumer_key else "no-key",
)


# --------------------------------------------------
# CSV FALLBACK
# --------------------------------------------------

with st.sidebar.expander(
    "Upload CSV instead",
    expanded=False,
):

    upload = st.file_uploader(
        "Daily OHLCV CSV",
        type=["csv"],
        help="Required columns: date, open, high, low, close, volume",
    )

    load_csv = st.button(
        "Load uploaded CSV",
        key="load_csv",
        use_container_width=True,
    )

    if load_csv:

        if upload is None:
            st.warning("Choose a CSV file first.")

        else:
            try:
                uploaded = pd.read_csv(upload)

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

                    st.session_state["active_symbol"] = (
                        symbol or "UPLOADED"
                    )

                    st.session_state["data_source"] = "Uploaded CSV"
                    st.session_state["data_error"] = None

                    st.session_state["auto_fetch_signature"] = (
                        request_signature
                    )

                    clear_analysis_results()

                    st.success("CSV loaded.")

            except Exception as exc:
                st.error(f"Could not load CSV: {exc}")


if refresh:
    fetch_market_history_cached.clear()
    st.session_state["auto_fetch_signature"] = None

st.sidebar.caption(
    "Data loads automatically when the symbol or period changes. "
    "Kotak data is cached for 30 minutes."
)


# --------------------------------------------------
# AUTOMATIC DATA FETCH
# --------------------------------------------------

if (
    st.session_state["auto_fetch_signature"] != request_signature
    and symbol
):

    st.session_state["auto_fetch_signature"] = request_signature

    st.session_state["price_data"] = None
    st.session_state["active_symbol"] = None
    st.session_state["data_source"] = None

    clear_analysis_results()

    if has_kotak_key:

        try:
            with st.spinner(
                f"Loading {symbol} history from Kotak Neo..."
            ):

                prices = fetch_market_history_cached(
                    symbol,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    request_signature[2],
                )

            st.session_state["price_data"] = prices
            st.session_state["active_symbol"] = symbol
            st.session_state["data_source"] = "Kotak Neo"
            st.session_state["data_error"] = None

        except Exception as exc:
            st.session_state["data_error"] = str(exc)

    else:

        st.session_state["data_error"] = (
            "Automatic Kotak data is not configured. "
            "Add NEO_CONSUMER_KEY in Streamlit app Settings → Secrets, "
            "or upload a CSV in the sidebar."
        )


# --------------------------------------------------
# HEADER AND DATA CHECK
# --------------------------------------------------

st.title("📈 StockVision AI")

st.caption(
    "Indian equities · Technical analysis · AI forecasts · "
    "Fundamental research"
)

price_df = st.session_state["price_data"]

active_symbol = (
    st.session_state["active_symbol"] or symbol
)

if price_df is None:

    if st.session_state["data_error"]:
        st.warning(st.session_state["data_error"])

    st.info(
        "Choose an NSE symbol in the sidebar. With Kotak credentials "
        "configured, StockVision AI fetches its history automatically."
    )

    st.markdown(
        """
        **Enable automatic data on Streamlit Cloud**

        Open **Manage app → Settings → Secrets** and add:

        ```toml
        NEO_CONSUMER_KEY = "your_Kotak_Neo_consumer_key"
        NEO_ENVIRONMENT = "prod"
        ```

        Save the secrets and reboot the app. Never commit API keys to GitHub.
        """
    )

    st.stop()


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

try:
    feature_df = build_features(price_df)

except Exception as exc:
    st.error(f"Could not prepare market data: {exc}")
    st.stop()

if feature_df.empty:
    st.error("No valid price records are available.")
    st.stop()

latest = feature_df.iloc[-1]

last_price = float(latest["close"])

previous_price = (
    float(feature_df["close"].iloc[-2])
    if len(feature_df) > 1
    else last_price
)

change_pct = (
    ((last_price / previous_price) - 1) * 100
    if previous_price
    else 0
)

rsi_value = latest.get("rsi_14", float("nan"))

latest_date = pd.to_datetime(
    latest["date"]
).date()


# --------------------------------------------------
# MARKET SUMMARY
# --------------------------------------------------

st.caption(
    f"**{active_symbol}** · {st.session_state['data_source']} · "
    f"Latest candle: {latest_date:%d %b %Y} · "
    f"{len(feature_df):,} daily records"
)

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Latest close",
    f"₹{last_price:,.2f}",
    f"{change_pct:+.2f}%",
)

m2.metric(
    "RSI (14)",
    f"{rsi_value:.1f}" if pd.notna(rsi_value) else "N/A",
)

m3.metric(
    "Daily records",
    f"{len(feature_df):,}",
)

m4.metric(
    "Latest volume",
    f"{float(latest['volume']):,.0f}",
)

st.divider()


# --------------------------------------------------
# FOUR MAIN SECTIONS
# --------------------------------------------------

overview_tab, forecast_tab, research_tab, watchlist_tab = st.tabs(
    [
        "📊 Overview",
        "🤖 Forecast Lab",
        "🔬 Research",
        "⭐ Watchlist",
    ]
)


# ==================================================
# OVERVIEW
# ==================================================

with overview_tab:

    st.subheader("Price trend")

    chart_columns = ["close"] + [
        column
        for column in ("sma_20", "sma_50", "sma_200")
        if column in feature_df.columns
        and feature_df[column].notna().any()
    ]

    price_chart = px.line(
        feature_df,
        x="date",
        y=chart_columns,
        labels={
            "date": "Date",
            "value": "Price (₹)",
            "variable": "",
        },
    )

    price_chart.update_layout(
        height=430,
        hovermode="x unified",
        margin=dict(l=5, r=5, t=15, b=5),
        legend_title_text="",
    )

    st.plotly_chart(
        price_chart,
        use_container_width=True,
    )

    left, right = st.columns([1, 2])

    with left:

        st.subheader("Technical signal")

        try:
            signal = generate_technical_signal(feature_df)

            st.metric(
                "Technical bias",
                signal["overall_signal"],
            )

            st.metric(
                "Signal score",
                f"{signal['score']:+.0f} / 100",
            )

            st.caption(
                "Rule-based indicator summary, not a probability "
                "or buy/sell recommendation."
            )

        except ValueError:
            st.info(
                f"The combined signal needs a valid 200-day average. "
                f"Loaded history: {len(feature_df)} daily records."
            )

    with right:

        st.subheader("Momentum")

        c1, c2 = st.columns(2)

        rsi_chart = px.line(
            feature_df,
            x="date",
            y="rsi_14",
        )

        rsi_chart.add_hline(
            y=70,
            line_dash="dash",
        )

        rsi_chart.add_hline(
            y=30,
            line_dash="dash",
        )

        rsi_chart.update_layout(
            height=260,
            margin=dict(l=3, r=3, t=10, b=5),
        )

        macd_chart = px.line(
            feature_df,
            x="date",
            y=["macd", "macd_signal"],
        )

        macd_chart.update_layout(
            height=260,
            margin=dict(l=3, r=3, t=10, b=5),
        )

        c1.plotly_chart(
            rsi_chart,
            use_container_width=True,
        )

        c2.plotly_chart(
            macd_chart,
            use_container_width=True,
        )

    with st.expander("Recent indicator readings"):

        columns = [
            column
            for column in (
                "date",
                "close",
                "sma_20",
                "sma_50",
                "sma_200",
                "rsi_14",
                "macd",
                "atr_14",
                "volatility_20d",
                "volume_ratio",
            )
            if column in feature_df.columns
        ]

        st.dataframe(
            feature_df[columns]
            .tail(30)
            .sort_values("date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Download historical data"):

        st.download_button(
            "Download CSV",
            data=feature_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{active_symbol}_stockvision.csv",
            mime="text/csv",
        )


# ==================================================
# FORECAST LAB
# ==================================================

with forecast_tab:

    st.subheader("AI forecasts")

    st.caption(
        "XGBoost estimates forward log returns. Forecasts are "
        "experimental, not guaranteed future prices."
    )

    horizons = st.multiselect(
        "Forecast horizons",
        FORECAST_HORIZONS,
        default=[21, 63],
        format_func=lambda x: f"{x} trading days",
        key="forecast_horizons",
    )

    if len(feature_df) < 800:

        st.warning(
            f"Only {len(feature_df)} trading records are loaded. "
            "Longer horizons need more history after indicator warm-up."
        )

    if st.button(
        "Train selected models",
        type="primary",
    ):

        if not horizons:
            st.warning("Select at least one forecast horizon.")

        else:

            try:
                with st.spinner("Training XGBoost models..."):

                    trained = train_forecast_models(
                        feature_df,
                        horizons=horizons,
                    )

                    predicted = predict_latest(
                        feature_df,
                        trained,
                    )

                st.session_state["trained_models"] = trained
                st.session_state["forecasts"] = predicted

            except Exception as exc:
                st.error(f"Model training failed: {exc}")

    trained = st.session_state["trained_models"]
    predictions = st.session_state["forecasts"]

    if trained and predictions is not None:

        display = predictions.copy()

        display["Horizon"] = (
            display["horizon_days"].astype(str) + " days"
        )

        display = display.rename(
            columns={
                "current_price": "Current price (₹)",
                "predicted_price": "Estimated price (₹)",
                "predicted_return_pct": "Estimated return (%)",
            }
        )

        st.dataframe(
            display[
                [
                    "Horizon",
                    "Current price (₹)",
                    "Estimated price (₹)",
                    "Estimated return (%)",
                ]
            ].style.format({
                "Current price (₹)": "₹{:,.2f}",
                "Estimated price (₹)": "₹{:,.2f}",
                "Estimated return (%)": "{:+.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        evaluation = pd.DataFrame([
            {
                "Horizon": f"{h} days",
                "MAE": result["mae"],
                "RMSE": result["rmse"],
                "Directional accuracy": (
                    result["directional_accuracy"] * 100
                ),
                "Validation rows": result["validation_rows"],
            }
            for h, result in trained.items()
        ])

        st.subheader("Chronological holdout evaluation")

        st.dataframe(
            evaluation.style.format({
                "MAE": "{:.5f}",
                "RMSE": "{:.5f}",
                "Directional accuracy": "{:.1f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Walk-forward backtesting"):

        backtest_horizons = st.multiselect(
            "Backtest horizons",
            FORECAST_HORIZONS,
            default=[21, 63],
            format_func=lambda x: f"{x} trading days",
            key="backtest_horizons",
        )

        max_folds = st.slider(
            "Maximum test periods per horizon",
            3,
            20,
            8,
        )

        if st.button("Run backtest"):

            if not backtest_horizons:
                st.warning("Select a horizon.")

            else:

                try:
                    with st.spinner("Running walk-forward backtest..."):

                        summary, details = walk_forward_backtest(
                            feature_df,
                            horizons=backtest_horizons,
                            min_train_rows=200,
                            max_folds=max_folds,
                        )

                    st.session_state["backtest_summary"] = summary
                    st.session_state["backtest_predictions"] = details

                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")

        if st.session_state["backtest_summary"] is not None:

            st.dataframe(
                st.session_state["backtest_summary"],
                use_container_width=True,
                hide_index=True,
            )

        if st.session_state["backtest_predictions"] is not None:

            st.dataframe(
                st.session_state["backtest_predictions"],
                use_container_width=True,
                hide_index=True,
            )


# ==================================================
# RESEARCH
# ==================================================

with research_tab:

    st.subheader("Research tools")

    # --------------------------------------------------
    # VALUATION
    # --------------------------------------------------

    with st.expander(
        "Long-term valuation scenarios",
        expanded=True,
    ):

        st.caption(
            "Illustrative EPS growth and future P/E assumptions."
        )

        a, b, c = st.columns(3)

        eps = a.number_input(
            "Current annual EPS (₹)",
            0.01,
            value=10.0,
            key="val_eps",
        )

        years = b.selectbox(
            "Forecast period",
            [3, 5],
            index=1,
        )

        valuation_price = c.number_input(
            "Current price (₹)",
            0.01,
            value=float(last_price),
            key="val_price",
        )

        a, b, c = st.columns(3)

        bear_growth = a.number_input(
            "Bear EPS growth (%)",
            value=5.0,
        )

        bear_pe = a.number_input(
            "Bear future P/E",
            min_value=0.1,
            value=15.0,
        )

        base_growth = b.number_input(
            "Base EPS growth (%)",
            value=12.0,
        )

        base_pe = b.number_input(
            "Base future P/E",
            min_value=0.1,
            value=20.0,
        )

        bull_growth = c.number_input(
            "Bull EPS growth (%)",
            value=18.0,
        )

        bull_pe = c.number_input(
            "Bull future P/E",
            min_value=0.1,
            value=25.0,
        )

        if st.button("Calculate scenarios"):

            assumptions = {
                "Bear": {
                    "eps_growth": bear_growth / 100,
                    "future_pe": bear_pe,
                },
                "Base": {
                    "eps_growth": base_growth / 100,
                    "future_pe": base_pe,
                },
                "Bull": {
                    "eps_growth": bull_growth / 100,
                    "future_pe": bull_pe,
                },
            }

            try:
                st.session_state["valuation_results"] = (
                    calculate_long_term_scenarios(
                        current_price=valuation_price,
                        current_eps=eps,
                        years=years,
                        scenarios=assumptions,
                    )
                )

            except Exception as exc:
                st.error(f"Valuation failed: {exc}")

        if st.session_state["valuation_results"] is not None:

            st.dataframe(
                st.session_state["valuation_results"],
                use_container_width=True,
                hide_index=True,
            )

    # --------------------------------------------------
    # RISK
    # --------------------------------------------------

    with st.expander("Historical risk analysis"):

        risk_rate = st.number_input(
            "Annual risk-free rate (%)",
            0.0,
            30.0,
            6.0,
            0.25,
        )

        confidence = st.selectbox(
            "VaR confidence",
            [0.90, 0.95, 0.99],
            index=1,
        )

        if st.button("Calculate risk metrics"):

            try:
                st.session_state["risk_results"] = (
                    calculate_risk_metrics(
                        feature_df,
                        risk_free_rate=risk_rate / 100,
                        confidence_level=confidence,
                    )
                )

            except Exception as exc:
                st.error(f"Risk calculation failed: {exc}")

        risk = st.session_state["risk_results"]

        if risk is not None:

            cols = st.columns(4)

            cols[0].metric(
                "Annual volatility",
                f"{risk['annualized_volatility_pct']:.2f}%",
            )

            cols[1].metric(
                "Maximum drawdown",
                f"{risk['maximum_drawdown_pct']:.2f}%",
            )

            cols[2].metric(
                "Sharpe ratio",
                f"{risk['sharpe_ratio']:.2f}",
            )

            cols[3].metric(
                "Daily VaR",
                f"{risk['daily_var_pct']:.2f}%",
            )

            st.caption(
                "Historical estimates do not define maximum future losses."
            )

    # --------------------------------------------------
    # FUNDAMENTALS
    # --------------------------------------------------

    with st.expander("Fundamental analysis"):

        financial_file = st.file_uploader(
            "Annual financial statements CSV",
            type=["csv"],
            key="fundamentals_upload",
        )

        st.caption(
            "Required columns: fiscal_year, revenue, operating_profit, "
            "net_profit, eps, total_equity, total_debt, current_assets, "
            "current_liabilities. Use consistent units and consolidated statements."
        )

        if st.button("Analyse fundamentals"):

            if financial_file is None:
                st.warning("Upload a financial statements CSV first.")

            else:

                try:
                    financials = pd.read_csv(financial_file)

                    result = analyze_fundamentals(
                        financials,
                        market_price=last_price,
                    )

                    st.session_state["fundamental_results"] = result
                    st.session_state["fundamental_symbol"] = active_symbol

                except Exception as exc:
                    st.error(f"Fundamental analysis failed: {exc}")

        fundamentals = st.session_state["fundamental_results"]

        if (
            fundamentals is not None
            and st.session_state["fundamental_symbol"] == active_symbol
        ):

            st.dataframe(
                pd.DataFrame([
                    {"Metric": k, "Value": v}
                    for k, v in fundamentals.items()
                ]),
                use_container_width=True,
                hide_index=True,
            )


# ==================================================
# WATCHLIST AND SCANNER
# ==================================================

with watchlist_tab:

    st.subheader("NSE watchlist")

    with st.form(
        "watchlist_add_form",
        clear_on_submit=True,
    ):

        new_symbol = st.text_input(
            "Add stock symbol",
            placeholder="Example: TCS",
        )

        add_clicked = st.form_submit_button(
            "Add to watchlist"
        )

    if add_clicked:

        try:
            result = add_symbol(new_symbol)

            if result["success"]:
                st.success(result["message"])
            else:
                st.info(result["message"])

        except Exception as exc:
            st.error(f"Could not add symbol: {exc}")

    watchlist = load_watchlist()

    if watchlist.empty:

        st.info("Your watchlist is empty.")

    else:

        st.dataframe(
            watchlist,
            use_container_width=True,
            hide_index=True,
        )

        remove_choice = st.selectbox(
            "Remove a symbol",
            [""] + watchlist["symbol"].tolist(),
            key="remove_choice",
        )

        if st.button("Remove selected symbol"):

            if remove_choice:

                try:
                    remove_symbol(remove_choice)
                    st.rerun()

                except Exception as exc:
                    st.error(str(exc))

        st.divider()

        st.subheader("Watchlist scanner")

        scan_days = st.selectbox(
            "Scanner history",
            [500, 750, 1000],
            index=1,
            format_func=lambda x: f"{x} calendar days",
        )

        if st.button(
            "Scan watchlist",
            type="primary",
        ):

            try:
                with st.spinner(
                    "Looking up tokens and scanning stocks..."
                ):

                    client = create_neo_client()

                    update_instruments(
                        watchlist["symbol"].tolist(),
                        client=client,
                    )

                    results = scan_watchlist(
                        watchlist,
                        lookback_days=scan_days,
                        client=client,
                    )

                st.session_state["scanner_results"] = results

            except Exception as exc:
                st.error(f"Scanner failed: {exc}")

        results = st.session_state["scanner_results"]

        if results is not None:

            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download scan results",
                results.to_csv(index=False).encode("utf-8"),
                file_name="stockvision_watchlist_scan.csv",
                mime="text/csv",
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "StockVision AI · Experimental financial research software · "
    "Market data may be delayed. Forecasts and valuation scenarios are uncertain."
)

