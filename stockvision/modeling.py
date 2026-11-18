
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


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

FORECAST_HORIZONS = [21, 63, 126, 252]


# --------------------------------------------------
# MODEL FEATURES
# --------------------------------------------------

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

MACD_SOURCE_COLUMNS = [
    "macd",
    "macd_signal",
    "macd_histogram",
]


# --------------------------------------------------
# PREPARE MODEL DATA
# --------------------------------------------------

def prepare_model_data(feature_df):
    """
    Prepare technical features and future log-return targets.

    Target:
        log(future closing price / current closing price)

    This avoids training directly on absolute stock-price changes.
    """

    if not isinstance(feature_df, pd.DataFrame):
        raise TypeError(
            "feature_df must be a pandas DataFrame."
        )

    required_columns = {
        "date",
        "close",
        *MACD_SOURCE_COLUMNS,
        *[c for c in FEATURE_COLUMNS if c not in {
            "macd_percent",
            "macd_signal_percent",
            "macd_histogram_percent",
        }],
    }

    missing = required_columns - set(feature_df.columns)

    if missing:
        raise ValueError(
            "Missing model columns. Run build_features() first. "
            "Missing: " + ", ".join(sorted(missing))
        )

    df = feature_df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    numeric_columns = list(
        set(FEATURE_COLUMNS + MACD_SOURCE_COLUMNS + ["close"])
    )

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = (
        df.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates(
            subset="date",
            keep="last"
        )
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "No valid price data is available."
        )

    if (df["close"] <= 0).any():
        raise ValueError(
            "Closing prices must be positive."
        )

    # Percentage-scaled MACD features.
    df["macd_percent"] = df["macd"] / df["close"]

    df["macd_signal_percent"] = (
        df["macd_signal"] / df["close"]
    )

    df["macd_histogram_percent"] = (
        df["macd_histogram"] / df["close"]
    )

    # Replace infinite values with missing values.
    df[FEATURE_COLUMNS] = (
        df[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
    )

    # Preserve chronological row positions.
    df["_position"] = np.arange(len(df))

    return df, FEATURE_COLUMNS.copy()


# --------------------------------------------------
# CREATE XGBOOST MODEL
# --------------------------------------------------

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


# --------------------------------------------------
# TRAIN ONE FORECAST HORIZON
# --------------------------------------------------

def train_one_horizon(
    df,
    feature_columns,
    horizon,
    test_size=0.20,
    model_dir=MODEL_DIR,
):
    """
    Train and evaluate one forecasting horizon.

    Uses chronological holdout validation.

    Training samples whose future target overlaps
    the validation period are excluded.

    A separate final model is fitted on all labelled
    observations for the latest forecast.
    """

    horizon = int(horizon)

    if horizon <= 0:
        raise ValueError(
            "Forecast horizon must be positive."
        )

    if not 0 < test_size < 0.5:
        raise ValueError(
            "test_size must be greater than 0 and less than 0.5."
        )

    working = df.copy().reset_index(drop=True)

    if "_position" not in working.columns:
        working["_position"] = np.arange(len(working))

    target_column = f"target_{horizon}d"

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

    if horizon >= total_rows:
        raise ValueError(
            f"Not enough observations for a {horizon}-day forecast."
        )

    # --------------------------------------------------
    # CHRONOLOGICAL VALIDATION SPLIT
    # --------------------------------------------------

    # Reserve rows for the validation period.
    # Also reserve the future horizon needed to calculate
    # actual validation targets.
    validation_rows = max(
        20,
        int(total_rows * test_size)
    )

    maximum_validation_rows = (
        total_rows - horizon - 1
    )

    if maximum_validation_rows < 20:
        raise ValueError(
            f"Insufficient historical data for the {horizon}-day model. "
            "Collect more daily price history."
        )

    validation_rows = min(
        validation_rows,
        maximum_validation_rows
    )

    if validation_rows < 20:
        raise ValueError(
            f"Insufficient validation observations "
            f"for the {horizon}-day model."
        )

    # Move the training boundary earlier to account for
    # the future target period.
    split_position = (
        total_rows - horizon - validation_rows
    )

    # --------------------------------------------------
    # PURGE OVERLAPPING TRAINING TARGETS
    # --------------------------------------------------

    train_mask = (
        (working["_position"] + horizon <= split_position)
        & working[target_column].notna()
    )

    test_mask = (
        (working["_position"] >= split_position)
        & (working["_position"] < split_position + validation_rows)
        & working[target_column].notna()
    )

    train_data = working.loc[train_mask].dropna(
        subset=feature_columns
    )

    test_data = working.loc[test_mask].dropna(
        subset=feature_columns
    )

    if len(train_data) < 100:
        raise ValueError(
            f"Insufficient training data for the {horizon}-day model. "
            f"Available training rows: {len(train_data)}. "
            "Collect more historical data."
        )

    if len(test_data) < 20:
        raise ValueError(
            f"Insufficient validation data for the {horizon}-day model. "
            f"Available test rows: {len(test_data)}."
        )

    # --------------------------------------------------
    # VALIDATION TRAINING
    # --------------------------------------------------

    X_train = train_data[feature_columns]
    y_train = train_data[target_column]

    X_test = test_data[feature_columns]
    y_test = test_data[target_column]

    validation_model = create_model()

    validation_model.fit(
        X_train,
        y_train
    )

    predictions = validation_model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    # Directional accuracy.
    actual_direction = np.sign(
        y_test.to_numpy()
    )

    predicted_direction = np.sign(
        predictions
    )

    directional_accuracy = np.mean(
        actual_direction == predicted_direction
    )

    # --------------------------------------------------
    # FINAL MODEL TRAINING
    # --------------------------------------------------

    # Include every observation whose future target
    # is already known in the current historical dataset.
    labelled_data = working.dropna(
        subset=feature_columns + [target_column]
    )

    final_model = create_model()

    final_model.fit(
        labelled_data[feature_columns],
        labelled_data[target_column]
    )

    # --------------------------------------------------
    # SAVE TRAINED MODEL
    # --------------------------------------------------

    model_dir = Path(model_dir)

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        model_dir / f"xgboost_{horizon}d.joblib"
    )

    model_bundle = {
        "model": final_model,
        "horizon": horizon,
        "feature_columns": feature_columns,
        "trained_at": datetime.now().astimezone().isoformat(),
    }

    joblib.dump(
        model_bundle,
        model_path
    )

    return {
        "model": final_model,
        "horizon": horizon,
        "feature_columns": feature_columns,
        "model_path": str(model_path),
        "mae": float(mae),
        "rmse": float(rmse),
        "directional_accuracy": float(
            directional_accuracy
        ),
        "training_rows": len(labelled_data),
        "validation_rows": len(test_data),
    }


# --------------------------------------------------
# TRAIN ALL FORECAST MODELS
# --------------------------------------------------

def train_forecast_models(
    feature_df,
    horizons=FORECAST_HORIZONS,
    test_size=0.20,
    model_dir=MODEL_DIR,
):
    """
    Train separate XGBoost models for each horizon.

    Default horizons:
        21 trading days
        63 trading days
        126 trading days
        252 trading days

    Returns a dictionary of trained models and metrics.
    """

    df, feature_columns = prepare_model_data(
        feature_df
    )

    trained_models = {}

    for horizon in horizons:

        horizon = int(horizon)

        if horizon <= 0:
            raise ValueError(
                "Forecast horizons must be positive."
            )

        print(
            f"Training {horizon}-trading-day model..."
        )

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


# --------------------------------------------------
# GENERATE LATEST FORECASTS
# --------------------------------------------------

def predict_latest(
    feature_df,
    trained_models,
):
    """
    Generate estimated future prices using the latest
    available technical feature row.

    Returns a DataFrame containing:
        horizon_days
        current_price
        predicted_return_pct
        predicted_price
    """

    if not trained_models:
        raise ValueError(
            "No trained forecasting models were supplied."
        )

    df, _ = prepare_model_data(feature_df)

    latest = df.iloc[[-1]]

    current_price = float(
        latest["close"].iloc[0]
    )

    forecasts = []

    for horizon, bundle in trained_models.items():

        model = bundle["model"]

        feature_columns = bundle["feature_columns"]

        missing = [
            column
            for column in feature_columns
            if column not in latest.columns
        ]

        if missing:
            raise ValueError(
                "Forecast features are missing: "
                + ", ".join(missing)
            )

        X_latest = latest[feature_columns]

        if X_latest.isna().any().any():
            raise ValueError(
                "Latest row contains missing model features. "
                "Collect more historical data."
            )

        predicted_log_return = float(
            model.predict(X_latest)[0]
        )

        if not np.isfinite(predicted_log_return):
            raise ValueError(
                f"Model returned an invalid prediction "
                f"for the {horizon}-day horizon."
            )

        predicted_return = (
            np.exp(predicted_log_return) - 1
        )

        predicted_price = (
            current_price * np.exp(predicted_log_return)
        )

        if not np.isfinite(predicted_price):
            raise ValueError(
                f"Predicted price is outside the valid numerical "
                f"range for the {horizon}-day horizon."
            )

        forecasts.append({
            "horizon_days": int(horizon),
            "current_price": current_price,
            "predicted_return_pct": (
                predicted_return * 100
            ),
            "predicted_price": float(predicted_price),
        })

    return (
        pd.DataFrame(forecasts)
        .sort_values("horizon_days")
        .reset_index(drop=True)
    )


# --------------------------------------------------
# LOAD SAVED MODEL
# --------------------------------------------------

def load_saved_model(
    horizon,
    model_dir=MODEL_DIR,
):
    """Load a previously saved forecasting model bundle."""

    horizon = int(horizon)

    model_path = (
        Path(model_dir) / f"xgboost_{horizon}d.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {model_path}"
        )

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(
            "Saved model file has an unexpected format. "
            "Retrain the model."
        )

    return bundle
