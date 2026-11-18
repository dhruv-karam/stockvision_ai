
import hashlib
import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st


# ==================================================
# IMPORT PROJECT MODULES
# ==================================================

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

from stockvision.valuation import (
    calculate_long_term_scenarios,
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

from stockvision.fundamentals import (
    fetch_auto_fundamentals,
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="StockVision AI",
    page_icon="📈",
    layout="wide",
)


# ==================================================
# CUSTOM DESIGN
# ==================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1440px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(125, 135, 150, 0.08);
        border: 1px solid rgba(150, 150, 150, 0.18);
        border-radius: 12px;
        padding: 13px;
    }

    h1 {
        letter-spacing: -0.03em;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
        padding: 0.65rem 1rem;
    }

    div[data-testid="stExpander"] {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# STREAMLIT CLOUD SECRETS
# ==================================================

def apply_streamlit_secrets():
    """Load Kotak credentials from Streamlit Cloud Secrets."""

    try:
        consumer_key = st.secrets.get(
            "NEO_CONSUMER_KEY",
            ""
        )

        environment = st.secrets.get(
            "NEO_ENVIRONMENT",
            "prod"
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


apply_streamlit_secrets()

consumer_key = os.getenv(
    "NEO_CONSUMER_KEY",
    ""
).strip()

has_kotak_key = bool(
    consumer_key
    and not consumer_key.startswith("YOUR_")
)


# ==================================================
# SESSION STATE
# ==================================================

DEFAULT_STATE = {
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
    "valuation_symbol": None,
    "valuation_years": None,

    "risk_results": None,

    "fundamental_results": None,
    "fundamental_symbol": None,

    "scanner_results": None,
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


def clear_analysis_results():
    """Clear results belonging to an old dataset."""

    keys = [
        "trained_models",
        "forecasts",
        "backtest_summary",
        "backtest_predictions",
        "valuation_results",
        "valuation_symbol",
        "valuation_years",
        "risk_results",
        "fundamental_results",
        "fundamental_symbol",
        "scanner_results",
    ]

    for key in keys:
        st.session_state[key] = None


# ==================================================
# CACHED HISTORICAL DATA
# ==================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def fetch_market_history_cached(
    symbol,
    start_text,
    end_text,
    credential_fingerprint,
):
    
@st.cache_data(
    ttl=86400,
    show_spinner=False,
)
def fetch_fundamentals_cached(
    symbol,
    market_price,
    forecast_years=5,
):
    """
    Automatically retrieve and calculate company fundamentals.

    Cache for 24 hours to reduce external API requests.
    """

    return fetch_auto_fundamentals(
        symbol=symbol,
        market_price=market_price,
        forecast_years=forecast_years,
    )

    """
    Fetch historical prices using Kotak Neo.

    Results are cached for 30 minutes to avoid
    unnecessary repeated historical API requests.
    """

    # Credential fingerprint forms part of the cache key.
    _ = credential_fingerprint

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
            f"Instrument lookup failed: {errors}"
        )

    prices = fetch_daily_history(
        symbol=symbol,
        start_date=start_text,
        end_date=end_text,
        client=client,
    )

    if prices.empty:
        raise RuntimeError(
            f"No historical candles received for {symbol}."
        )

    return prices


# ==================================================
# HEADER
# ==================================================

st.title("📈 StockVision AI")

st.caption(
    "Indian equities · Technical analysis · AI forecasting "
    "· Fundamental research"
)


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("⚙️ Market Settings")

today = date.today()

if "selected_symbol" not in st.session_state:
    st.session_state["selected_symbol"] = "RELIANCE"

symbol = st.sidebar.text_input(
    "NSE stock symbol",
    key="selected_symbol",
    help="Examples: RELIANCE, TCS, INFY, HDFCBANK",
).strip().upper()

period_options = {
    "1 year": 365,
    "3 years": 1095,
    "5 years": 1825,
}

period_label = st.sidebar.selectbox(
    "Historical data range",
    list(period_options.keys()),
    index=1,
)

period_days = period_options[period_label]

start_date = today - timedelta(
    days=period_days
)

end_date = today

refresh_button = st.sidebar.button(
    "↻ Refresh market data",
    use_container_width=True,
)


# ==================================================
# REQUEST SIGNATURE
# ==================================================

credential_fingerprint = (
    hashlib.sha256(
        consumer_key.encode()
    ).hexdigest()[:16]
    if consumer_key
    else "no-key"
)

request_signature = (
    f"{symbol}|{period_days}|{credential_fingerprint}"
)


# ==================================================
# CSV UPLOAD FALLBACK
# ==================================================

with st.sidebar.expander(
    "📂 Upload historical CSV",
    expanded=False,
):

    uploaded_file = st.file_uploader(
        "Upload OHLCV data",
        type=["csv"],
        key="market_csv_upload",
    )

    load_csv_button = st.button(
        "Load uploaded CSV",
        use_container_width=True,
    )

    if load_csv_button:

        if uploaded_file is None:
            st.warning("Please upload a CSV file.")

        else:

            try:

                uploaded_df = pd.read_csv(
                    uploaded_file
                )

                required_columns = [
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]

                missing = [
                    column
                    for column in required_columns
                    if column not in uploaded_df.columns
                ]

                if missing:

                    st.error(
                        "Missing columns: "
                        + ", ".join(missing)
                    )

                else:

                    st.session_state["price_data"] = (
                        uploaded_df[required_columns].copy()
                    )

                    st.session_state["active_symbol"] = (
                        symbol or "UPLOADED"
                    )

                    st.session_state["data_source"] = (
                        "Uploaded CSV"
                    )

                    st.session_state["data_error"] = None

                    st.session_state["auto_fetch_signature"] = (
                        request_signature
                    )

                    clear_analysis_results()

                    st.success("Historical CSV loaded.")

            except Exception as exc:

                st.error(
                    f"CSV loading failed: {exc}"
                )


st.sidebar.divider()

st.sidebar.caption(
    "Kotak historical data is cached for 30 minutes. "
    "A manual refresh bypasses the cache."
)


# ==================================================
# MANUAL REFRESH
# ==================================================

if refresh_button:

    fetch_market_history_cached.clear()

    st.session_state["auto_fetch_signature"] = None


# ==================================================
# AUTOMATIC DATA DOWNLOAD
# ==================================================

if (
    symbol
    and st.session_state["auto_fetch_signature"]
    != request_signature
):

    st.session_state["auto_fetch_signature"] = (
        request_signature
    )

    st.session_state["price_data"] = None

    st.session_state["active_symbol"] = None

    st.session_state["data_source"] = None

    st.session_state["data_error"] = None

    clear_analysis_results()

    if has_kotak_key:

        try:

            with st.spinner(
                f"Fetching {symbol} historical data from Kotak Neo..."
            ):

                prices = fetch_market_history_cached(
                    symbol,
                    start_date.isoformat(),
                    end_date.isoformat(),
                    credential_fingerprint,
                )

            st.session_state["price_data"] = prices

            st.session_state["active_symbol"] = symbol

            st.session_state["data_source"] = "Kotak Neo"

            st.session_state["data_error"] = None

        except Exception as exc:

            st.session_state["data_error"] = str(exc)

    else:

        st.session_state["data_error"] = (
            "Kotak Neo credentials are not configured. "
            "Add NEO_CONSUMER_KEY in Streamlit Secrets "
            "or upload historical CSV data."
        )


# ==================================================
# MARKET DATA STATUS
# ==================================================

price_df = st.session_state["price_data"]

active_symbol = (
    st.session_state["active_symbol"]
    or symbol
)

if price_df is None:

    st.warning(
        st.session_state["data_error"]
        or "Historical market data is not loaded."
    )

    st.info(
        "Choose a stock in the sidebar. StockVision AI "
        "will fetch its history automatically when Kotak "
        "Neo credentials and API access are available."
    )

    st.markdown(
        """
        **Automatic data configuration**

        Go to Manage app → Settings → Secrets and add:

        ```toml
        NEO_CONSUMER_KEY = "YOUR_ACTUAL_CONSUMER_KEY"
        NEO_ENVIRONMENT = "prod"
        ```

        Never commit your actual credentials to GitHub.
        """
    )

    st.stop()


# ==================================================
# FEATURE ENGINEERING
# ==================================================

try:

    feature_df = build_features(
        price_df
    )

except Exception as exc:

    st.error(
        f"Technical feature calculation failed: {exc}"
    )

    st.stop()


if feature_df.empty:

    st.error(
        "No valid historical market data is available."
    )

    st.stop()


# ==================================================
# LATEST MARKET VALUES
# ==================================================

latest = feature_df.iloc[-1]

latest_price = float(
    latest["close"]
)

previous_price = (
    float(feature_df["close"].iloc[-2])
    if len(feature_df) > 1
    else latest_price
)

daily_change_pct = (
    (latest_price / previous_price - 1) * 100
    if previous_price > 0
    else 0
)

latest_rsi = latest.get(
    "rsi_14",
    float("nan")
)

latest_volume = float(
    latest["volume"]
)

latest_date = pd.to_datetime(
    latest["date"]
).date()


# ==================================================
# MARKET SUMMARY
# ==================================================

st.caption(
    f"**{active_symbol}** · "
    f"{st.session_state['data_source']} · "
    f"Latest candle: {latest_date:%d %b %Y} · "
    f"{len(feature_df):,} daily records"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Latest Close",
    f"₹{latest_price:,.2f}",
    f"{daily_change_pct:+.2f}%",
)

col2.metric(
    "RSI (14)",
    (
        f"{latest_rsi:.2f}"
        if pd.notna(latest_rsi)
        else "N/A"
    ),
)

col3.metric(
    "Historical Records",
    f"{len(feature_df):,}",
)

col4.metric(
    "Trading Volume",
    f"{latest_volume:,.0f}",
)

st.divider()


# ==================================================
# MAIN NAVIGATION
# ==================================================

overview_tab, forecast_tab, research_tab, watchlist_tab = st.tabs(
    [
        "📊 Overview",
        "🤖 Forecast Lab",
        "🔬 Research",
        "⭐ Watchlist",
    ]
)


# ==================================================
# SECTION 1 — OVERVIEW
# ==================================================

with overview_tab:

    st.subheader("Price Performance")

    available_price_columns = [
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

    price_chart = px.line(
        feature_df,
        x="date",
        y=available_price_columns,
        labels={
            "date": "Date",
            "value": "Price (₹)",
            "variable": "Indicator",
        },
    )

    price_chart.update_layout(
        height=420,
        hovermode="x unified",
        margin=dict(
            l=5,
            r=5,
            t=15,
            b=5,
        ),
        legend_title_text="",
    )

    st.plotly_chart(
        price_chart,
        use_container_width=True,
    )

    left, right = st.columns([1, 2])

    with left:

        st.subheader("Technical Signal")

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
                f"{signal['score']:+.0f} / 100",
            )

            st.metric(
                "RSI Condition",
                signal["rsi_status"],
            )

            st.caption(
                "A rule-based technical summary, not a "
                "buy/sell recommendation."
            )

        except ValueError as exc:

            st.info(
                f"Complete signal unavailable: {exc}"
            )

    with right:

        st.subheader("Momentum Analysis")

        rsi_chart = px.line(
            feature_df,
            x="date",
            y="rsi_14",
            labels={
                "rsi_14": "RSI",
                "date": "Date",
            },
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
            margin=dict(
                l=5,
                r=5,
                t=10,
                b=5,
            ),
        )

        st.plotly_chart(
            rsi_chart,
            use_container_width=True,
        )

    st.subheader("MACD")

    macd_chart = px.line(
        feature_df,
        x="date",
        y=[
            "macd",
            "macd_signal",
        ],
        labels={
            "date": "Date",
            "value": "MACD",
            "variable": "Indicator",
        },
    )

    macd_chart.update_layout(
        height=300,
        margin=dict(
            l=5,
            r=5,
            t=10,
            b=5,
        ),
    )

    st.plotly_chart(
        macd_chart,
        use_container_width=True,
    )

    with st.expander("Recent technical readings"):

        columns = [
            column
            for column in [
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
            ]
            if column in feature_df.columns
        ]

        st.dataframe(
            feature_df[columns]
            .tail(30)
            .sort_values(
                "date",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Download historical data"):

        st.download_button(
            "Download CSV",
            data=feature_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name=f"{active_symbol}_historical.csv",
            mime="text/csv",
        )


# ==================================================
# SECTION 2 — FORECAST LAB
# ==================================================

with forecast_tab:

    st.subheader("AI Forecasting")

    st.caption(
        "XGBoost models estimate forward returns over selected "
        "trading-day horizons. Predictions are experimental."
    )

    selected_horizons = st.multiselect(
        "Select forecast horizons",
        options=FORECAST_HORIZONS,
        default=[21, 63],
        format_func=lambda x: f"{x} trading days",
        key="forecast_horizons",
    )

    st.info(
        f"Available historical records: {len(feature_df)}"
    )

    if len(feature_df) < 500:

        st.warning(
            "The dataset may be too small for reliable longer-horizon "
            "training. Consider collecting more historical data."
        )

    if st.button(
        "🚀 Train Forecast Models",
        type="primary",
    ):

        if not selected_horizons:

            st.warning(
                "Please select at least one forecast horizon."
            )

        else:

            try:

                with st.spinner(
                    "Training forecasting models..."
                ):

                    trained = train_forecast_models(
                        feature_df=feature_df,
                        horizons=selected_horizons,
                    )

                    forecasts = predict_latest(
                        feature_df=feature_df,
                        trained_models=trained,
                    )

                st.session_state["trained_models"] = trained

                st.session_state["forecasts"] = forecasts

                st.success(
                    "Forecasting completed successfully."
                )

            except Exception as exc:

                st.error(
                    f"Forecast training failed: {exc}"
                )

    trained_models = st.session_state["trained_models"]

    forecasts = st.session_state["forecasts"]

    if (
        trained_models is not None
        and forecasts is not None
    ):

        st.divider()

        st.subheader("Forecast Results")

        display = forecasts.copy()

        display["Horizon"] = (
            display["horizon_days"].astype(str)
            + " trading days"
        )

        display = display.rename(
            columns={
                "current_price": "Current Price (₹)",
                "predicted_price": "Estimated Price (₹)",
                "predicted_return_pct": "Estimated Return (%)",
            }
        )

        st.dataframe(
            display[
                [
                    "Horizon",
                    "Current Price (₹)",
                    "Estimated Price (₹)",
                    "Estimated Return (%)",
                ]
            ].style.format({
                "Current Price (₹)": "₹{:,.2f}",
                "Estimated Price (₹)": "₹{:,.2f}",
                "Estimated Return (%)": "{:+.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Model Evaluation")

        evaluation_rows = []

        for horizon, result in trained_models.items():

            evaluation_rows.append({
                "Horizon": f"{horizon} days",
                "MAE": result["mae"],
                "RMSE": result["rmse"],
                "Directional Accuracy": (
                    result["directional_accuracy"] * 100
                ),
                "Validation Rows": result["validation_rows"],
            })

        evaluation = pd.DataFrame(
            evaluation_rows
        )

        st.dataframe(
            evaluation.style.format({
                "MAE": "{:.5f}",
                "RMSE": "{:.5f}",
                "Directional Accuracy": "{:.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Historical holdout performance is not a guarantee "
            "of future accuracy."
        )

    # --------------------------------------------------
    # WALK-FORWARD BACKTEST
    # --------------------------------------------------

    with st.expander(
        "🧪 Walk-Forward Backtesting"
    ):

        st.caption(
            "Evaluate historical predictions using chronological "
            "training and testing periods."
        )

        backtest_horizons = st.multiselect(
            "Backtest horizons",
            options=FORECAST_HORIZONS,
            default=[21, 63],
            format_func=lambda x: f"{x} trading days",
            key="backtest_horizons",
        )

        max_folds = st.slider(
            "Maximum historical test periods",
            min_value=3,
            max_value=20,
            value=8,
        )

        if st.button("Run Backtest"):

            if not backtest_horizons:

                st.warning(
                    "Select at least one horizon."
                )

            else:

                try:

                    with st.spinner(
                        "Running historical backtest..."
                    ):

                        summary, details = walk_forward_backtest(
                            feature_df=feature_df,
                            horizons=backtest_horizons,
                            min_train_rows=200,
                            max_folds=max_folds,
                        )

                    st.session_state["backtest_summary"] = summary

                    st.session_state["backtest_predictions"] = details

                    st.success("Backtest completed.")

                except Exception as exc:

                    st.error(
                        f"Backtest failed: {exc}"
                    )

        summary = st.session_state["backtest_summary"]

        details = st.session_state["backtest_predictions"]

        if summary is not None:

            st.subheader("Backtest Summary")

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True,
            )

        if details is not None:

            st.subheader("Historical Forecasts")

            st.dataframe(
                details,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Backtest CSV",
                data=details.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=f"{active_symbol}_backtest.csv",
                mime="text/csv",
            )


# ==================================================
# SECTION 3 — RESEARCH
# ==================================================

with research_tab:

    st.subheader("Research Tools")

    # ==================================================
    # VALUATION
    # ==================================================

    with st.expander(
        "💰 Long-Term Valuation Scenarios",
        expanded=True,
    ):

        st.caption(
            "Enter your assumptions. StockVision AI will calculate "
            "the Bear, Base and Bull scenarios."
        )

        with st.form("valuation_form"):

            col1, col2, col3 = st.columns(3)

            with col1:

                current_eps = st.number_input(
                    "Current Annual EPS (₹)",
                    min_value=0.0,
                    value=0.0,
                    step=0.5,
                    help="Enter verified annual or trailing twelve-month EPS.",
                )

            with col2:

                valuation_years = st.selectbox(
                    "Forecast Period",
                    options=[3, 5],
                    index=1,
                    format_func=lambda x: f"{x} years",
                )

            with col3:

                valuation_price = st.number_input(
                    "Current Share Price (₹)",
                    min_value=0.01,
                    value=float(latest_price),
                    step=1.0,
                )

            st.markdown(
                "#### Scenario Assumptions"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                bear_growth = st.number_input(
                    "Bear EPS Growth (%)",
                    value=5.0,
                    min_value=-99.0,
                    max_value=100.0,
                )

                bear_pe = st.number_input(
                    "Bear Future P/E",
                    value=15.0,
                    min_value=0.1,
                )

            with col2:

                base_growth = st.number_input(
                    "Base EPS Growth (%)",
                    value=12.0,
                    min_value=-99.0,
                    max_value=100.0,
                )

                base_pe = st.number_input(
                    "Base Future P/E",
                    value=20.0,
                    min_value=0.1,
                )

            with col3:

                bull_growth = st.number_input(
                    "Bull EPS Growth (%)",
                    value=18.0,
                    min_value=-99.0,
                    max_value=100.0,
                )

                bull_pe = st.number_input(
                    "Bull Future P/E",
                    value=25.0,
                    min_value=0.1,
                )

            calculate_valuation = st.form_submit_button(
                "Calculate Scenarios",
                use_container_width=True,
                type="primary",
            )

        if calculate_valuation:

            if current_eps <= 0:

                st.error(
                    "Please enter the company's actual positive EPS."
                )

            else:

                scenario_inputs = {
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

                    valuation_df = (
                        calculate_long_term_scenarios(
                            current_price=valuation_price,
                            current_eps=current_eps,
                            years=valuation_years,
                            scenarios=scenario_inputs,
                        )
                    )

                    st.session_state["valuation_results"] = (
                        valuation_df
                    )

                    st.session_state["valuation_symbol"] = (
                        active_symbol
                    )

                    st.session_state["valuation_years"] = (
                        valuation_years
                    )

                except Exception as exc:

                    st.error(
                        f"Valuation calculation failed: {exc}"
                    )

        results = st.session_state["valuation_results"]

        result_symbol = st.session_state["valuation_symbol"]

        result_years = st.session_state["valuation_years"]

        if (
            results is not None
            and result_symbol == active_symbol
        ):

            st.divider()

            st.subheader(
                "Your Valuation Results"
            )

            st.caption(
                f"{active_symbol} · {result_years}-year scenario analysis"
            )

            bear = results[
                results["scenario"] == "Bear"
            ]

            base = results[
                results["scenario"] == "Base"
            ]

            bull = results[
                results["scenario"] == "Bull"
            ]

            col1, col2, col3 = st.columns(3)

            scenario_rows = [
                (col1, bear, "🐻 Bear"),
                (col2, base, "⚖️ Base"),
                (col3, bull, "🚀 Bull"),
            ]

            for column, scenario, heading in scenario_rows:

                if scenario.empty:
                    continue

                row = scenario.iloc[0]

                with column:

                    st.markdown(
                        f"#### {heading}"
                    )

                    st.metric(
                        "Estimated Future Price",
                        f"₹{row['estimated_future_price']:,.2f}",
                        (
                            f"{row['total_price_return_pct']:+.2f}% "
                            "total return"
                        ),
                    )

                    st.caption(
                        f"Annualized return: "
                        f"{row['annualized_price_return_pct']:+.2f}%"
                    )

            st.caption(
                "Assumption-driven estimates, not guaranteed targets "
                "or intrinsic fair values. Dividends, dilution and "
                "corporate actions are excluded."
            )

    # ==================================================
    # RISK ANALYSIS
    # ==================================================

    with st.expander(
        "⚠️ Historical Risk Analysis"
    ):

        st.caption(
            "Measure historical volatility, drawdown and "
            "risk-adjusted returns."
        )

        risk_rate = st.number_input(
            "Annual Risk-Free Rate (%)",
            min_value=0.0,
            max_value=30.0,
            value=6.0,
            step=0.25,
        )

        confidence = st.selectbox(
            "VaR Confidence Level",
            options=[0.90, 0.95, 0.99],
            index=1,
            format_func=lambda x: f"{x:.0%}",
        )

        if st.button(
            "Calculate Risk Metrics"
        ):

            try:

                st.session_state["risk_results"] = (
                    calculate_risk_metrics(
                        feature_df,
                        risk_free_rate=risk_rate / 100,
                        confidence_level=confidence,
                    )
                )

            except Exception as exc:

                st.error(
                    f"Risk calculation failed: {exc}"
                )

        risk = st.session_state["risk_results"]

        if risk is not None:

            a, b, c, d = st.columns(4)

            a.metric(
                "Annual Volatility",
                f"{risk['annualized_volatility_pct']:.2f}%",
            )

            b.metric(
                "Maximum Drawdown",
                f"{risk['maximum_drawdown_pct']:.2f}%",
            )

            c.metric(
                "Sharpe Ratio",
                f"{risk['sharpe_ratio']:.2f}",
            )

            d.metric(
                "Daily VaR",
                f"{risk['daily_var_pct']:.2f}%",
            )

            st.caption(
                "Historical risk estimates are not guarantees "
                "of future losses."
            )


    # ==================================================
    # AUTOMATIC FUNDAMENTAL ANALYSIS
    # ==================================================

    with st.expander(
        "🏢 Automatic Fundamental Analysis",
        expanded=False,
    ):

        st.caption(
            "StockVision AI automatically retrieves financial statements "
            "and calculates company fundamentals."
        )

        try:

            with st.spinner(
                f"Analysing {active_symbol} financial statements..."
            ):

                fundamental_data = fetch_fundamentals_cached(
                    symbol=active_symbol,
                    market_price=latest_price,
                    forecast_years=5,
                )

            metrics = fundamental_data["metrics"]
            assumptions = fundamental_data["assumptions"]
            scenarios = fundamental_data["scenarios"]

            def display_value(value, suffix="%", digits=2):

                if value is None or pd.isna(value):
                    return "N/A"

                return f"{float(value):.{digits}f}{suffix}"

            st.markdown(
                f"### {fundamental_data['company']}"
            )

            st.caption(
                f"{active_symbol} · "
                f"Latest financial year: "
                f"{fundamental_data['latest_fiscal_year']}"
            )

            st.divider()

            # ------------------------------------------
            # VALUATION METRICS
            # ------------------------------------------

            st.subheader("Company Valuation")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Current EPS",
                (
                    f"₹{metrics['eps']:,.2f}"
                    if pd.notna(metrics["eps"])
                    else "N/A"
                ),
            )

            col2.metric(
                "Current P/E",
                display_value(
                    metrics["pe_ratio"],
                    suffix="x",
                ),
            )

            col3.metric(
                "Revenue CAGR",
                display_value(
                    metrics["revenue_growth_pct"]
                ),
            )

            col4.metric(
                "EPS CAGR",
                display_value(
                    metrics["eps_growth_pct"]
                ),
            )

            # ------------------------------------------
            # PROFITABILITY
            # ------------------------------------------

            st.subheader("Profitability")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "ROE",
                display_value(metrics["roe_pct"]),
            )

            col2.metric(
                "ROCE",
                display_value(metrics["roce_pct"]),
            )

            col3.metric(
                "Operating Margin",
                display_value(
                    metrics["operating_margin_pct"]
                ),
            )

            col4.metric(
                "Net Profit Margin",
                display_value(
                    metrics["net_profit_margin_pct"]
                ),
            )

            # ------------------------------------------
            # FINANCIAL STRENGTH
            # ------------------------------------------

            st.subheader("Financial Strength")

            col1, col2 = st.columns(2)

            col1.metric(
                "Debt / Equity",
                display_value(
                    metrics["debt_to_equity"],
                    suffix="x",
                ),
            )

            col2.metric(
                "Current Ratio",
                display_value(
                    metrics["current_ratio"],
                    suffix="x",
                ),
            )

            # ------------------------------------------
            # AUTOMATIC GROWTH ASSUMPTIONS
            # ------------------------------------------

            st.divider()

            st.subheader("Five-Year Growth Assessment")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Bear Growth",
                display_value(
                    assumptions["bear_growth"]
                ),
            )

            col2.metric(
                "Base Growth",
                display_value(
                    assumptions["base_growth"]
                ),
            )

            col3.metric(
                "Bull Growth",
                display_value(
                    assumptions["bull_growth"]
                ),
            )

            st.caption(
                "Growth assumptions are derived from historical "
                f"financial performance. Method: "
                f"{assumptions['growth_method']}."
            )

            # ------------------------------------------
            # AUTOMATIC VALUATION SCENARIOS
            # ------------------------------------------

            st.subheader("Estimated Five-Year Share Price")

            scenario_columns = st.columns(3)

            scenario_details = [
                ("Bear", "🐻"),
                ("Base", "⚖️"),
                ("Bull", "🚀"),
            ]

            for column, (scenario_name, emoji) in zip(
                scenario_columns,
                scenario_details,
            ):

                row = scenarios[
                    scenarios["scenario"] == scenario_name
                ]

                if row.empty:
                    continue

                result = row.iloc[0]

                with column:

                    st.markdown(
                        f"#### {emoji} {scenario_name}"
                    )

                    st.metric(
                        "Estimated Future Price",
                        f"₹{result['estimated_future_price']:,.2f}",
                        (
                            f"{result['total_price_return_pct']:+.2f}%"
                        ),
                    )

                    st.caption(
                        "Annualized return: "
                        + display_value(
                            result["annualized_price_return_pct"]
                        )
                    )

            st.caption(
                "Future P/E assumptions are automatically derived "
                "from the current P/E with scenario adjustments. "
                "These are estimates, not guaranteed targets."
            )

            # ------------------------------------------
            # FINANCIAL HISTORY
            # ------------------------------------------

            history = fundamental_data["financial_history"]

            if history is not None and not history.empty:

                st.divider()

                with st.expander(
                    "View reported annual financial history"
                ):

                    st.dataframe(
                        history,
                        use_container_width=True,
                        hide_index=True,
                    )

            st.caption(
                f"Data source: {fundamental_data['source']} · "
                f"Updated: {fundamental_data['updated_at']}"
            )

        except Exception as exc:

            st.warning(
                "Automatic fundamental analysis is currently "
                "unavailable for this stock."
            )

            st.caption(
                f"Data source response: {exc}"
            )

            st.caption(
                "No financial assumptions have been fabricated. "
                "Try again later or verify the company financial statements."
            )


# ==================================================
# SECTION 4 — WATCHLIST
# ==================================================

with watchlist_tab:

    st.subheader("⭐ NSE Watchlist")

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
            "Your watchlist is empty. Add your first stock above."
        )

    else:

        st.metric(
            "Saved Stocks",
            len(watchlist),
        )

        st.dataframe(
            watchlist,
            use_container_width=True,
            hide_index=True,
        )

        selected_remove = st.selectbox(
            "Select stock to remove",
            options=[""] + watchlist["symbol"].tolist(),
            key="watchlist_remove",
        )

        if st.button(
            "Remove Selected Stock"
        ):

            if selected_remove:

                try:

                    remove_symbol(selected_remove)

                    st.session_state["scanner_results"] = None

                    st.success(
                        f"{selected_remove} removed."
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(str(exc))

        # --------------------------------------------------
        # WATCHLIST SCANNER
        # --------------------------------------------------

        with st.expander(
            "🔍 Scan all watchlist stocks"
        ):

            lookback = st.selectbox(
                "Scanner history",
                [500, 750, 1000],
                index=1,
                format_func=lambda x: f"{x} calendar days",
            )

            if st.button(
                "Run Watchlist Scanner",
                type="primary",
            ):

                try:

                    with st.spinner(
                        "Searching instruments and analysing stocks..."
                    ):

                        client = create_neo_client()

                        update_instruments(
                            watchlist["symbol"].tolist(),
                            client=client,
                        )

                        scanner_results = scan_watchlist(
                            watchlist_df=watchlist,
                            lookback_days=lookback,
                            client=client,
                        )

                    st.session_state["scanner_results"] = (
                        scanner_results
                    )

                    st.success(
                        "Watchlist scan completed."
                    )

                except Exception as exc:

                    st.error(
                        f"Scanner failed: {exc}"
                    )

            results = st.session_state["scanner_results"]

            if results is not None:

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
                    file_name="stockvision_scanner.csv",
                    mime="text/csv",
                )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "StockVision AI · Experimental financial research software · "
    "Forecasts are uncertain and are not investment advice."
)
