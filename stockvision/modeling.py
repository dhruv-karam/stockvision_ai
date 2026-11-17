from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

FORECAST_HORIZONS = [21, 63, 126, 252]

# Use features that are less dependent on a stock's absolute price.
FEATURE_COLUMNS = [
    "daily_return",
    "log_return",
    "return_5d",
    "return_10d",
    "return_21d",
    "rsi_14",
    "bb_width",
    "bb_position",
    "atr_percent",
    "volatility_20d",
    "volatility_60d",
    "volume_ratio",
    "price_vs_sma_20",
    "price_vs_sma_50",
    "price_vs_sma_100",
    "price_vs_sma_200",
    "macd_percent",
    "macd_signal_percent",
    "macd_histogram_percent",
]


def prepare_model_data(feature_df):
    """
    Prepare features and future log-return targets.

    The target for each horizon is:

        log(future closing price / current closing price)

    This avoids training directly on absolute stock-price changes.
    """

    required_columns = {"date", "close"}

    missing = required_columns - set(feature_df.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = feature_df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = (
        df.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates(subset="date", keep="last")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("No valid price data is available.")

    if (df["close"] <= 0).any():
        raise ValueError("Closing prices must be positive.")

    # Keep the original row position for time-series validation.
    df["_position"] = np.arange(len(df))

    # Add percentage-scaled MACD features.
    df["macd_percent"] = df["macd"] / df["close"]
    df["macd_signal_percent"] = (
        df["macd_signal"] / df["close"]
    )
    df["macd_histogram_percent"] = (
        df["macd_histogram"] / df["close"]
    )

    available_features = [
        column
        for column in FEATURE_COLUMNS
        if column in df.columns
    ]

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features. Run build_features() first. "
            "Missing: " + ", ".join(missing_features)
        )

    # Replace invalid numerical values.
    df[available_features] = (
        df[available_features]
        .replace([np.inf, -np.inf], np.nan)
    )

    return df, available_features


def create_model():
    """Create an XGBoost regression model."""

    return XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )


def train_one_horizon(
    df,
    feature_columns,
    horizon,
    test_size=0.20,
    model_dir=MODEL_DIR,
):
    """
    Train and evaluate one forecasting horizon.

    Validation is chronological, not randomly shuffled.

    Training rows whose future target would overlap the test
    period are excluded from the training set.
    """

    target_column = f"target_{horizon}d"

    working = df.copy()

    # Future log return over the requested horizon.
    working[target_column] = np.log(
        working["close"].shift(-horizon)
        / working["close"]
    )

    total_rows = len(working)

    if total_rows < 150:
        raise ValueError(
            f"Not enough historical data for the {horizon}-day model. "
            "Collect more daily price history."
        )

    split_position = int(total_rows * (1 - test_size))

    # Purge training examples whose target reaches into the test period.
    train_mask = (
        (working["_position"] + horizon <= split_position)
        & working[target_column].notna()
    )

    test_mask = (
        (working["_position"] >= split_position)
        & working[target_column].notna()
    )

    # Keep only rows with complete model features.
    train_data = working.loc[train_mask].dropna(
        subset=feature_columns
    )

    test_data = working.loc[test_mask].dropna(
        subset=feature_columns
    )

    if len(train_data) < 100:
        raise ValueError(
            f"Insufficient training data for the {horizon}-day model. "
            f"Available training rows: {len(train_data)}."
        )

    if len(test_data) < 20:
        raise ValueError(
            f"Insufficient validation data for the {horizon}-day model. "
            f"Available test rows: {len(test_data)}."
        )

    X_train = train_data[feature_columns]
    y_train = train_data[target_column]

    X_test = test_data[feature_columns]
    y_test = test_data[target_column]

    # Train a model for out-of-sample evaluation.
    validation_model = create_model()
    validation_model.fit(X_train, y_train)

    predictions = validation_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    # Directional accuracy: did the model predict the right sign?
    actual_direction = np.sign(y_test.to_numpy())
    predicted_direction = np.sign(predictions)

    directional_accuracy = np.mean(
        actual_direction == predicted_direction
    )

    # Refit on all available labelled observations for forecasting.
    labelled_data = working.dropna(
        subset=feature_columns + [target_column]
    )

    final_model = create_model()

    final_model.fit(
        labelled_data[feature_columns],
        labelled_data[target_column]
    )

    # Save the model and its metadata.
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"xgboost_{horizon}d.joblib"

    model_bundle = {
        "model": final_model,
        "horizon": horizon,
        "feature_columns": feature_columns,
        "trained_at": datetime.now().isoformat(),
    }

    joblib.dump(model_bundle, model_path)

    return {
        "model": final_model,
        "horizon": horizon,
        "feature_columns": feature_columns,
        "model_path": str(model_path),
        "mae": float(mae),
        "rmse": float(rmse),
        "directional_accuracy": float(directional_accuracy),
        "training_rows": len(labelled_data),
        "validation_rows": len(test_data),
    }


def train_forecast_models(
    feature_df,
    horizons=FORECAST_HORIZONS,
    test_size=0.20,
    model_dir=MODEL_DIR,
):
    """
    Train one XGBoost model for every requested horizon.

    Returns a dictionary containing models and evaluation metrics.
    """

    df, feature_columns = prepare_model_data(feature_df)

    trained_models = {}

    for horizon in horizons:
        horizon = int(horizon)

        if horizon <= 0:
            raise ValueError("Forecast horizons must be positive.")

        print(f"Training {horizon}-trading-day model...")

        result = train_one_horizon(
            df=df,
            feature_columns=feature_columns,
            horizon=horizon,
            test_size=test_size,
            model_dir=model_dir,
        )

        trained_models[horizon] = result

        print(
            f"Completed {horizon}-day model | "
            f"MAE: {result['mae']:.5f} | "
            f"RMSE: {result['rmse']:.5f} | "
            f"Directional accuracy: "
            f"{result['directional_accuracy']:.2%}"
        )

    return trained_models


def predict_latest(feature_df, trained_models):
    """
    Generate forecasts using the latest available feature row.

    Returns predicted percentage returns and estimated prices.
    """

    df, _ = prepare_model_data(feature_df)

    latest = df.iloc[[-1]]
    current_price = float(latest["close"].iloc[0])

    forecasts = []

    for horizon, bundle in trained_models.items():

        model = bundle["model"]
        feature_columns = bundle["feature_columns"]

        X_latest = latest[feature_columns]

        if X_latest.isna().any().any():
            raise ValueError(
                "The latest row has missing model features. "
                "Collect more historical data."
            )

        predicted_log_return = float(
            model.predict(X_latest)[0]
        )

        predicted_return = (
            np.exp(predicted_log_return) - 1
        )

        predicted_price = (
            current_price * np.exp(predicted_log_return)
        )

        forecasts.append({
            "horizon_days": int(horizon),
            "current_price": current_price,
            "predicted_return_pct": predicted_return * 100,
            "predicted_price": float(predicted_price),
        })

    return (
        pd.DataFrame(forecasts)
        .sort_values("horizon_days")
        .reset_index(drop=True)
    )


def load_saved_model(horizon, model_dir=MODEL_DIR):
    """Load a previously saved forecasting model."""

    model_path = (
        Path(model_dir) / f"xgboost_{int(horizon)}d.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {model_path}"
        )

    return joblib.load(model_path)
