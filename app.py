import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date, timedelta
from stockvision.signals import generate_technical_signal
from pathlib import Path
from stockvision.instruments import update_instruments
from stockvision.backtesting import walk_forward_backtest

from stockvision.data import fetch_daily_history
from stockvision.features import build_features
from stockvision.modeling import (
    train_forecast_models,
    predict_latest,
    FORECAST_HORIZONS,
)
from stockvision.valuation import calculate_long_term_scenarios
from stockvision.watchlist import (
    load_watchlist,
    add_symbol,
    remove_symbol,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="StockVision AI",
    page_icon="📈",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "price_data" not in st.session_state:
    st.session_state.price_data = None

if "trained_models" not in st.session_state:
    st.session_state.trained_models = None

if "forecasts" not in st.session_state:
    st.session_state.forecasts = None

if "active_symbol" not in st.session_state:
    st.session_state.active_symbol = None


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📈 StockVision AI")

st.subheader(
    "Indian Stock Market Forecasting and Valuation Dashboard"
)

st.caption(
    "Historical market data • Machine learning forecasts "
    "• Technical indicators • Long-term valuation scenarios"
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Settings")

symbol = st.sidebar.text_input(
    "NSE Stock Symbol",
    value="RELIANCE",
).strip().upper()

if st.sidebar.button(
    "🔎 Find and Save NSE Token",
    use_container_width=True,
):
    if not symbol:
        st.sidebar.error("Enter a stock symbol first.")

    else:
        with st.spinner(
            f"Searching Kotak Neo for {symbol}..."
        ):
            try:
                result = update_instruments([symbol])

                if result["successful"]:
                    instrument = result["successful"][0]

                    st.sidebar.success(
                        f"{symbol} instrument saved!"
                    )

                    st.sidebar.code(
                        f"Exchange: {instrument['exchange_segment']}\n"
                        f"Token: {instrument['instrument_token']}"
                    )

                else:
                    for failure in result["failed"]:
                        st.sidebar.error(
                            f"{failure['symbol']}: {failure['error']}"
                        )

            except Exception as error:
                st.sidebar.error(
                    f"Instrument search failed: {error}"
                )

today = date.today()

start_date = st.sidebar.date_input(
    "Historical Data Start Date",
    value=today - timedelta(days=5 * 365),
    max_value=today,
)

end_date = st.sidebar.date_input(
    "Historical Data End Date",
    value=today,
    max_value=today,
)

st.sidebar.caption(
    "For the longer forecasting horizons, aim to collect "
    "at least 5 years of daily market history."
)

fetch_button = st.sidebar.button(
    "📥 Fetch Data from Kotak Neo",
    use_container_width=True,
)

st.sidebar.divider()

st.sidebar.caption(
    "StockVision AI is an experimental research tool. "
    "Forecasts are uncertain and are not investment advice."
)


# --------------------------------------------------
# FETCH MARKET DATA
# --------------------------------------------------

if fetch_button:

    if not symbol:
        st.sidebar.error("Enter a valid stock symbol.")

    elif start_date >= end_date:
        st.sidebar.error(
            "The end date must be later than the start date."
        )

    else:
        with st.spinner(
            f"Fetching {symbol} historical data from Kotak Neo..."
        ):
            try:
                price_df = fetch_daily_history(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )

                if price_df.empty:
                    st.error(
                        "No historical data was returned. "
                        "Check the instrument token and date range."
                    )

                else:
                    st.session_state.price_data = price_df
                    st.session_state.active_symbol = symbol
                    st.session_state.trained_models = None
                    st.session_state.forecasts = None

                    st.success(
                        f"Successfully fetched {len(price_df)} "
                        f"daily records for {symbol}!"
                    )

            except Exception as error:
                st.error(
                    f"Unable to fetch market data: {error}"
                )


# --------------------------------------------------
# CSV UPLOAD OPTION
# --------------------------------------------------

st.subheader("📂 Load Historical Data")

st.write(
    "You can also upload an OHLCV CSV file instead of "
    "fetching data directly from Kotak Neo."
)

uploaded_file = st.file_uploader(
    "Upload historical stock data",
    type=["csv"],
)

if uploaded_file is not None:

    if st.button("Load Uploaded CSV"):

        try:
            uploaded_df = pd.read_csv(uploaded_file)

            required_columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in uploaded_df.columns
            ]

            if missing_columns:
                st.error(
                    "CSV is missing required columns: "
                    + ", ".join(missing_columns)
                )

            else:
                st.session_state.price_data = uploaded_df[
                    required_columns
                ].copy()

                st.session_state.active_symbol = symbol
                st.session_state.trained_models = None
                st.session_state.forecasts = None

                st.success("CSV data loaded successfully!")

        except Exception as error:
            st.error(f"Could not read CSV: {error}")


st.divider()


# --------------------------------------------------
# CHECK LOADED DATA
# --------------------------------------------------

price_df = st.session_state.price_data

if price_df is None:

    st.info(
        "Start by fetching historical data from Kotak Neo "
        "or uploading a historical OHLCV CSV file."
    )

    st.subheader("What StockVision AI offers")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊")
        st.write("**Technical Analysis**")
        st.caption(
            "RSI, MACD, moving averages, Bollinger Bands "
            "and volatility."
        )

    with col2:
        st.markdown("### 🤖")
        st.write("**AI Forecasting**")
        st.caption(
            "XGBoost models for 21, 63, 126 and "
            "252 trading-day horizons."
        )

    with col3:
        st.markdown("### 💹")
        st.write("**Long-Term Valuation**")
        st.caption(
            "Bear, Base and Bull scenarios using EPS growth "
            "and future P/E assumptions."
        )

    st.stop()


# --------------------------------------------------
# PREPARE DATA
# --------------------------------------------------

try:
    price_df = build_features(price_df)

except Exception as error:
    st.error(f"Unable to process price data: {error}")
    st.stop()


active_symbol = st.session_state.active_symbol or symbol

latest_price = float(price_df["close"].iloc[-1])
previous_price = (
    float(price_df["close"].iloc[-2])
    if len(price_df) > 1
    else latest_price
)

price_change = latest_price - previous_price

price_change_pct = (
    price_change / previous_price * 100
    if previous_price != 0
    else 0
)


# --------------------------------------------------
# TOP LEVEL METRICS
# --------------------------------------------------

st.subheader(f"📌 {active_symbol} Market Overview")

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Latest Close",
    f"₹{latest_price:,.2f}",
    f"{price_change_pct:+.2f}%",
)

metric2.metric(
    "Historical Records",
    f"{len(price_df):,}",
)

metric3.metric(
    "Latest RSI",
    f"{price_df['rsi_14'].iloc[-1]:.2f}"
    if pd.notna(price_df["rsi_14"].iloc[-1])
    else "N/A",
)

metric4.metric(
    "Latest Trading Volume",
    f"{price_df['volume'].iloc[-1]:,.0f}",
)

st.caption(
    f"Latest available data date: "
    f"{price_df['date'].iloc[-1].date()}"
)

st.divider()


# --------------------------------------------------
# DASHBOARD TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📈 Market Analysis",
        "🤖 AI Forecasts",
        "💰 Long-Term Valuation",
        "📋 Historical Data",
        "🧪 Backtesting",
        "⭐ Watchlist",
    ]
)
# --------------------------------------------------
# TAB 1: MARKET ANALYSIS
# --------------------------------------------------

with tab1:
        # Technical Signal Summary

    st.subheader("🎯 Overall Technical Signal")

    try:
        signal = generate_technical_signal(price_df)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Technical Bias",
                signal["overall_signal"],
            )

        with col2:
            st.metric(
                "Technical Score",
                f"{signal['score']:+.1f} / 100",
            )

        with col3:
            st.metric(
                "RSI Status",
                signal["rsi_status"],
            )

        st.markdown("#### Individual Indicator Signals")

        st.dataframe(
            signal["indicators"],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Positive scores indicate bullish technical conditions; "
            "negative scores indicate bearish conditions. "
            "This is not a buy or sell recommendation."
        )

    except ValueError as error:
        st.info(
            f"Technical signal is not available yet: {error}"
        )

    st.divider()

    st.subheader("Historical Closing Price")

    price_chart = px.line(
        price_df,
        x="date",
        y="close",
        title=f"{active_symbol} Daily Closing Price",
    )

    price_chart.update_layout(
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        hovermode="x unified",
    )

    st.plotly_chart(
        price_chart,
        use_container_width=True,
    )

    st.subheader("Moving Average Analysis")

    moving_average_chart = px.line(
        price_df,
        x="date",
        y=["close", "sma_20", "sma_50", "sma_200"],
        title="Price and Moving Averages",
    )

    st.plotly_chart(
        moving_average_chart,
        use_container_width=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        rsi_chart = px.line(
            price_df,
            x="date",
            y="rsi_14",
            title="Relative Strength Index (RSI)",
        )

        rsi_chart.add_hline(
            y=70,
            line_dash="dash",
        )

        rsi_chart.add_hline(
            y=30,
            line_dash="dash",
        )

        st.plotly_chart(
            rsi_chart,
            use_container_width=True,
        )

    with col2:
        macd_chart = px.line(
            price_df,
            x="date",
            y=[
                "macd",
                "macd_signal",
            ],
            title="MACD Analysis",
        )

        st.plotly_chart(
            macd_chart,
            use_container_width=True,
        )

    st.subheader("Bollinger Bands")

    bollinger_chart = px.line(
        price_df,
        x="date",
        y=[
            "close",
            "bb_upper",
            "bb_middle",
            "bb_lower",
        ],
        title="Bollinger Band Analysis",
    )

    st.plotly_chart(
        bollinger_chart,
        use_container_width=True,
    )


# --------------------------------------------------
# TAB 2: AI FORECASTING
# --------------------------------------------------

with tab2:

    st.subheader("XGBoost Forecasting Engine")

    st.write(
        "Train separate models to estimate future log returns "
        "over different trading-day horizons."
    )

    selected_horizons = st.multiselect(
        "Select Forecast Horizons",
        options=FORECAST_HORIZONS,
        default=FORECAST_HORIZONS,
        format_func=lambda x: f"{x} Trading Days",
    )

    if len(price_df) < 300:
        st.warning(
            "Your dataset is relatively small. "
            "Longer forecast horizons may not have enough "
            "training and validation observations."
        )

    train_button = st.button(
        "🚀 Train Forecasting Models",
        type="primary",
    )

    if train_button:

        if not selected_horizons:
            st.warning("Select at least one forecast horizon.")

        else:
            with st.spinner(
                "Training XGBoost forecasting models..."
            ):

                try:
                    models = train_forecast_models(
                        feature_df=price_df,
                        horizons=selected_horizons,
                    )

                    forecasts = predict_latest(
                        feature_df=price_df,
                        trained_models=models,
                    )

                    st.session_state.trained_models = models
                    st.session_state.forecasts = forecasts

                    st.success(
                        "Forecasting models trained successfully!"
                    )

                except Exception as error:
                    st.error(
                        f"Model training failed: {error}"
                    )


    # Display saved results

    models = st.session_state.trained_models
    forecasts = st.session_state.forecasts

    if models is not None and forecasts is not None:

        st.divider()

        st.subheader("Forecast Results")

        display_forecasts = forecasts.copy()

        display_forecasts["horizon"] = (
            display_forecasts["horizon_days"].astype(str)
            + " trading days"
        )

        display_forecasts = display_forecasts.rename(
            columns={
                "current_price": "Current Price (₹)",
                "predicted_price": "Estimated Price (₹)",
                "predicted_return_pct": "Estimated Return (%)",
            }
        )

        st.dataframe(
            display_forecasts[
                [
                    "horizon",
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

        for horizon, result in models.items():

            evaluation_rows.append({
                "Horizon": f"{horizon} trading days",
                "MAE": result["mae"],
                "RMSE": result["rmse"],
                "Directional Accuracy (%)": (
                    result["directional_accuracy"] * 100
                ),
                "Training Rows": result["training_rows"],
                "Validation Rows": result["validation_rows"],
            })

        evaluation_df = pd.DataFrame(evaluation_rows)

        st.dataframe(
            evaluation_df.style.format({
                "MAE": "{:.5f}",
                "RMSE": "{:.5f}",
                "Directional Accuracy (%)": "{:.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Evaluation metrics are based on a chronological holdout "
            "validation split. They are historical measurements, "
            "not a guarantee of future performance."
        )

        forecast_chart = px.bar(
            forecasts,
            x="horizon_days",
            y="predicted_price",
            title="Estimated Future Prices",
            labels={
                "horizon_days": "Trading-Day Horizon",
                "predicted_price": "Estimated Price (₹)",
            },
        )

        st.plotly_chart(
            forecast_chart,
            use_container_width=True,
        )

    else:

        st.info(
            "No forecasting models have been trained yet. "
            "Select your horizons and click Train Forecasting Models."
        )


# --------------------------------------------------
# TAB 3: LONG-TERM VALUATION
# --------------------------------------------------

with tab3:

    st.subheader("Long-Term Fundamental Valuation")

    st.write(
        "Estimate possible future prices using current EPS, "
        "assumed earnings growth and future P/E multiples."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        valuation_price = st.number_input(
            "Current Share Price (₹)",
            min_value=0.01,
            value=float(latest_price),
            step=1.0,
        )

    with col2:
        current_eps = st.number_input(
            "Current Annual EPS (₹)",
            min_value=0.01,
            value=10.0,
            step=1.0,
        )

    with col3:
        valuation_years = st.selectbox(
            "Valuation Period",
            options=[3, 5],
            index=1,
            format_func=lambda x: f"{x} Years",
        )

    st.caption(
        "Enter the company's latest annual or trailing-twelve-month "
        "earnings per share. The default EPS is only a placeholder."
    )

    st.markdown("#### Scenario Assumptions")

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

    if st.button("Calculate Valuation Scenarios"):

        scenarios = {
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
            valuation_df = calculate_long_term_scenarios(
                current_price=valuation_price,
                current_eps=current_eps,
                years=valuation_years,
                scenarios=scenarios,
            )

            st.session_state.valuation_results = valuation_df

        except Exception as error:
            st.error(f"Valuation calculation failed: {error}")

    valuation_results = st.session_state.get(
        "valuation_results"
    )

    if valuation_results is not None:

        st.divider()

        st.subheader("Valuation Scenario Results")

        st.dataframe(
            valuation_results.style.format({
                "eps_growth_pct": "{:.2f}%",
                "future_pe": "{:.2f}",
                "estimated_future_eps": "₹{:.2f}",
                "estimated_future_price": "₹{:,.2f}",
                "total_price_return_pct": "{:+.2f}%",
                "annualized_price_return_pct": "{:+.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        valuation_chart = px.bar(
            valuation_results,
            x="scenario",
            y="estimated_future_price",
            title="Long-Term Estimated Price Scenarios",
            labels={
                "scenario": "Scenario",
                "estimated_future_price": "Estimated Price (₹)",
            },
        )

        st.plotly_chart(
            valuation_chart,
            use_container_width=True,
        )

        st.caption(
            "These are assumption-driven scenarios, not guaranteed "
            "targets or intrinsic fair values. Dividends, dilution, "
            "corporate actions and changes in financial quality "
            "are not included."
        )


# --------------------------------------------------
# TAB 4: HISTORICAL DATA
# --------------------------------------------------

with tab4:

    st.subheader("Historical Market Data")

    st.write(
        "Review the price history and technical indicators "
        "used by the forecasting engine."
    )

    st.dataframe(
        price_df.sort_values(
            "date",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = price_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Historical Data CSV",
        data=csv_data,
        file_name=f"{active_symbol}_historical_data.csv",
        mime="text/csv",
    )

# --------------------------------------------------
# TAB 5: WALK-FORWARD BACKTESTING
# --------------------------------------------------

with tab5:

    st.subheader("🧪 Historical Walk-Forward Backtesting")

    st.write(
        "Test historical forecasts using models trained only "
        "on information that would have been available "
        "at each forecast date."
    )

    backtest_horizons = st.multiselect(
        "Select backtesting horizons",
        options=FORECAST_HORIZONS,
        default=[21, 63],
        format_func=lambda x: f"{x} trading days",
        key="backtest_horizons",
    )

    max_folds = st.slider(
        "Maximum historical test periods per horizon",
        min_value=3,
        max_value=20,
        value=10,
    )

    run_backtest = st.button(
        "🚀 Run Walk-Forward Backtest",
        type="primary",
    )

    if run_backtest:

        if not backtest_horizons:
            st.warning("Select at least one horizon.")

        else:
            try:
                with st.spinner(
                    "Running historical model evaluations..."
                ):

                    summary_df, predictions_df = (
                        walk_forward_backtest(
                            feature_df=price_df,
                            horizons=backtest_horizons,
                            min_train_rows=200,
                            max_folds=max_folds,
                        )
                    )

                    st.session_state.backtest_summary = summary_df
                    st.session_state.backtest_predictions = (
                        predictions_df
                    )

                st.success("Backtesting completed!")

            except Exception as error:
                st.error(f"Backtesting failed: {error}")

    summary_df = st.session_state.get(
        "backtest_summary"
    )

    predictions_df = st.session_state.get(
        "backtest_predictions"
    )

    if summary_df is not None and predictions_df is not None:

        st.divider()

        st.subheader("Backtest Performance")

        display_summary = summary_df.copy()

        display_summary["Horizon"] = (
            display_summary["horizon_days"].astype(str)
            + " trading days"
        )

        st.dataframe(
            display_summary[
                [
                    "Horizon",
                    "number_of_forecasts",
                    "mae",
                    "rmse",
                    "baseline_mae",
                    "mae_improvement_pct",
                    "directional_accuracy_pct",
                ]
            ].style.format({
                "mae": "{:.5f}",
                "rmse": "{:.5f}",
                "baseline_mae": "{:.5f}",
                "mae_improvement_pct": "{:+.2f}%",
                "directional_accuracy_pct": "{:.2f}%",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "MAE and RMSE measure forward log-return errors. "
            "The baseline assumes zero forward return. "
            "Positive MAE improvement means the model had "
            "lower error than this baseline."
        )

        st.subheader("Predicted vs Actual Returns")

        comparison_chart = px.scatter(
            predictions_df,
            x="actual_return_pct",
            y="predicted_return_pct",
            color="horizon_days",
            hover_data=[
                "forecast_date",
                "target_date",
            ],
            title="Historical Forecast Comparison",
            labels={
                "actual_return_pct": "Actual Return (%)",
                "predicted_return_pct": "Predicted Return (%)",
                "horizon_days": "Horizon (Trading Days)",
            },
        )

        st.plotly_chart(
            comparison_chart,
            use_container_width=True,
        )

        st.subheader("Individual Historical Forecasts")

        st.dataframe(
            predictions_df,
            use_container_width=True,
            hide_index=True,
        )

        backtest_csv = predictions_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Backtest Results",
            data=backtest_csv,
            file_name=f"{active_symbol}_backtest.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "Run a backtest to see historical model performance."
        )
# --------------------------------------------------
# TAB 6: STOCK WATCHLIST
# --------------------------------------------------

with tab6:

    st.subheader("⭐ My NSE Stock Watchlist")

    st.write(
        "Maintain a personal list of NSE stocks you want "
        "to research and monitor."
    )

    # Add a stock
    with st.form("add_watchlist_form"):

        new_symbol = st.text_input(
            "Enter NSE stock symbol",
            placeholder="Example: TCS",
        )

        add_button = st.form_submit_button(
            "➕ Add Stock",
            use_container_width=True,
        )

    if add_button:

        try:
            result = add_symbol(new_symbol)

            if result["success"]:
                st.success(result["message"])
            else:
                st.warning(result["message"])

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(f"Unable to update watchlist: {error}")

    st.divider()

    # Display saved stocks
    watchlist_df = load_watchlist()

    st.subheader("Saved Stocks")

    if watchlist_df.empty:

        st.info(
            "Your watchlist is empty. Add your first stock above."
        )

    else:

        st.metric(
            "Stocks in Watchlist",
            len(watchlist_df),
        )

        st.dataframe(
            watchlist_df,
            use_container_width=True,
            hide_index=True,
        )

        # Remove a stock
        selected_symbol = st.selectbox(
            "Select a stock to remove",
            options=watchlist_df["symbol"].tolist(),
        )

        if st.button(
            "🗑️ Remove Selected Stock",
            use_container_width=True,
        ):

            try:
                result = remove_symbol(selected_symbol)

                if result["success"]:
                    st.success(result["message"])
                    st.rerun()
                else:
                    st.warning(result["message"])

            except Exception as error:
                st.error(
                    f"Unable to remove stock: {error}"
                )

    st.caption(
        "Your watchlist is saved locally in data/watchlist.csv. "
        "It is not automatically synchronized across devices."
    )
# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "StockVision AI | Experimental financial research software. "
    "Forecasts and valuation scenarios are uncertain. "
    "Always conduct independent research before making investment decisions."
)
