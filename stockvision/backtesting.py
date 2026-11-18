import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from stockvision.modeling import prepare_model_data


def walk_forward_backtest(
    feature_df,
    horizons=(21, 63, 126, 252),
    min_train_rows=200,
    max_folds=20,
):
    """
    Evaluate forecasting models using historical walk-forward testing.

    At each forecast date:
    1. Train only on observations whose future target was already known.
    2. Predict the forward return.
    3. Compare the prediction with the actual future return.

    Forecast origins are spaced by the forecast horizon, so the
    evaluation periods do not overlap for each individual horizon.

    Returns:
        summary_df: Performance metrics by forecast horizon.
        predictions_df: Individual historical forecasts.
    """

    df, feature_columns = prepare_model_data(feature_df)

    df = df.copy().reset_index(drop=True)
    df["_position"] = np.arange(len(df))

    total_rows = len(df)

    if total_rows < 300:
        raise ValueError(
            "At least 300 historical observations are recommended "
            "for walk-forward backtesting."
        )

    valid_features = df[feature_columns].notna().all(axis=1)

    valid_positions = np.flatnonzero(valid_features.to_numpy())

    if len(valid_positions) == 0:
        raise ValueError(
            "No rows contain complete model features. "
            "Collect more historical data."
        )

    first_valid_position = int(valid_positions[0])

    all_predictions = []
    summaries = []

    for horizon in horizons:

        horizon = int(horizon)

        if horizon <= 0:
            raise ValueError(
                "Forecast horizons must be positive."
            )

        target_column = f"target_{horizon}d"

        working = df.copy()

        working[target_column] = np.log(
            working["close"].shift(-horizon)
            / working["close"]
        )

        # The earliest test date must leave enough labelled,
        # fully populated training observations.
        first_origin = (
            first_valid_position
            + horizon
            + min_train_rows
            - 1
        )

        last_origin = total_rows - horizon - 1

        if first_origin > last_origin:
            continue

        origins = np.arange(
            first_origin,
            last_origin + 1,
            horizon,
        )

        if max_folds is not None and len(origins) > max_folds:
            selected_indices = np.linspace(
                0,
                len(origins) - 1,
                num=max_folds,
                dtype=int,
            )

            origins = origins[selected_indices]

        horizon_predictions = []

        for fold_number, origin in enumerate(origins, start=1):

            # Only use training targets that would have been
            # completely observable on the forecast date.
            train_mask = (
                working["_position"] + horizon <= origin
            )

            train_data = working.loc[train_mask].dropna(
                subset=feature_columns + [target_column]
            )

            test_row = working.iloc[[origin]]

            if (
                len(train_data) < min_train_rows
                or test_row[feature_columns].isna().any().any()
                or pd.isna(test_row[target_column].iloc[0])
            ):
                continue

            X_train = train_data[feature_columns]
            y_train = train_data[target_column]

            X_test = test_row[feature_columns]

            model = XGBRegressor(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
            )

            model.fit(X_train, y_train)

            predicted_log_return = float(
                model.predict(X_test)[0]
            )

            actual_log_return = float(
                test_row[target_column].iloc[0]
            )

            current_price = float(
                test_row["close"].iloc[0]
            )

            actual_future_price = current_price * np.exp(
                actual_log_return
            )

            predicted_future_price = current_price * np.exp(
                predicted_log_return
            )

            record = {
                "horizon_days": horizon,
                "fold": fold_number,
                "forecast_date": test_row["date"].iloc[0],
                "target_date": working["date"].iloc[
                    origin + horizon
                ],
                "current_price": current_price,
                "predicted_future_price": predicted_future_price,
                "actual_future_price": actual_future_price,
                "predicted_return_pct": (
                    np.exp(predicted_log_return) - 1
                ) * 100,
                "actual_return_pct": (
                    np.exp(actual_log_return) - 1
                ) * 100,
                "predicted_log_return": predicted_log_return,
                "actual_log_return": actual_log_return,
                "training_rows": len(train_data),
            }

            horizon_predictions.append(record)
            all_predictions.append(record)

        if not horizon_predictions:
            continue

        evaluation = pd.DataFrame(horizon_predictions)

        actual = evaluation["actual_log_return"].to_numpy()
        predicted = evaluation["predicted_log_return"].to_numpy()

        # Zero-return benchmark: assumes no change in price.
        zero_predictions = np.zeros(len(actual))

        model_mae = mean_absolute_error(actual, predicted)

        baseline_mae = mean_absolute_error(
            actual,
            zero_predictions,
        )

        directional_accuracy = np.mean(
            np.sign(actual) == np.sign(predicted)
        )

        summaries.append({
            "horizon_days": horizon,
            "number_of_forecasts": len(evaluation),
            "mae": model_mae,
            "rmse": np.sqrt(
                mean_squared_error(actual, predicted)
            ),
            "baseline_mae": baseline_mae,
            "mae_improvement_pct": (
                (baseline_mae - model_mae) / baseline_mae * 100
                if baseline_mae > 0
                else np.nan
            ),
            "directional_accuracy_pct": (
                directional_accuracy * 100
            ),
        })

    if not summaries:
        raise ValueError(
            "No valid backtest forecasts could be generated. "
            "You may need more historical data, especially "
            "for longer forecast horizons."
        )

    summary_df = pd.DataFrame(summaries)

    predictions_df = pd.DataFrame(all_predictions)

    return summary_df, predictions_df
