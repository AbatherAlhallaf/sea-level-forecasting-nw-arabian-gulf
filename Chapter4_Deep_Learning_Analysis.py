"""
Chapter 4: Deep-Learning Analysis of Sea-Level Variability

This script implements the deep-learning analyses presented in Chapter 4:

1. Develop univariate and multivariate CNN models.
2. Develop univariate and multivariate LSTM models.
3. Develop univariate and multivariate parallel CNN-LSTM models.
4. Evaluate out-of-sample predictions using MSE, RMSE, and MAE.
5. Estimate monthly residual-based and SHAP-based meteorological contributions.

The workflow uses a chronologically aligned dataset, training-only Min-Max
scaling, 10-day input sequences, and an 80/20 chronological split.

Required input file
-------------------
Place the following file in a folder named "data" beside this script:

data/Cleaned_Multi_data.csv
    Required columns: DATE, SLV (or WL1), AT (or AT2), AP (or P),
    WS, WD (or GD), and GS.

Random seeds and deterministic TensorFlow operations are configured to reduce
variation between runs. Exact results may still vary slightly across hardware
and software environments.

Results are saved in the "chapter4_outputs" folder.
"""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LSTM,
    MaxPooling1D,
    concatenate,
)


# =============================================================================
# 1. Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "data" / "Cleaned_Multi_data.csv"
OUTPUT_DIR = SCRIPT_DIR / "chapter4_outputs"
MODEL_DIR = OUTPUT_DIR / "models"

DATE_COLUMN = "DATE"
TARGET = "SLV"
METEOROLOGICAL = ["AT", "AP", "WS", "WD", "GS"]
MULTIVARIATE_INPUTS = [TARGET] + METEOROLOGICAL

SEQUENCE_LENGTH = 10
TRAIN_FRACTION = 0.80
RANDOM_SEED = 42

# Training configurations used in Chapter 4.
CNN_EPOCHS = 100
CNN_BATCH_SIZE = 64
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 128
HYBRID_EPOCHS = 100
HYBRID_BATCH_SIZE = 128

VALIDATION_FRACTION = 0.10
TRAIN_MODELS = True
RUN_SHAP = True
SAVE_PLOTS = True

# SHAP sampling configuration.
SHAP_BACKGROUND_SIZE = 100
SHAP_EXPLANATION_SIZE: int | None = None  # None explains the full test set.


# =============================================================================
# 2. Reproducibility and data preparation
# =============================================================================

def set_reproducibility(seed: int = RANDOM_SEED) -> None:
    """Set available random seeds and request deterministic TensorFlow ops."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def canonicalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Convert accepted alternative names to the Chapter 4 terminology."""
    data = data.copy()
    data.columns = [str(column).strip() for column in data.columns]
    aliases = {"WL1": "SLV", "AT2": "AT", "P": "AP", "GD": "WD"}
    rename_map = {
        old: new
        for old, new in aliases.items()
        if old in data.columns and new not in data.columns
    }
    return data.rename(columns=rename_map)


def load_aligned_data(path: Path) -> pd.DataFrame:
    """Load, validate, sort, and align the daily SLV/meteorological dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Place Cleaned_Multi_data.csv inside the data folder."
        )

    data = canonicalize_columns(pd.read_csv(path))
    required = [DATE_COLUMN] + MULTIVARIATE_INPUTS
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {missing}\n"
            f"Columns found: {data.columns.tolist()}"
        )

    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    for column in MULTIVARIATE_INPUTS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data[MULTIVARIATE_INPUTS] = data[MULTIVARIATE_INPUTS].replace(-999, np.nan)
    data = data.dropna(subset=required)
    data = data.sort_values(DATE_COLUMN)
    data = data.drop_duplicates(subset=[DATE_COLUMN], keep="first")
    data = data.set_index(DATE_COLUMN)

    if len(data) <= SEQUENCE_LENGTH + 10:
        raise ValueError("Too few aligned observations remain for sequence modeling.")
    return data[MULTIVARIATE_INPUTS]


def scale_and_sequence(data: pd.DataFrame) -> dict[str, object]:
    """
    Fit all scalers on chronological training rows and construct paired
    univariate/multivariate sequences with identical target dates.

    The first testing sequence may use preceding training observations as
    historical context, but its prediction target lies in the testing period.
    """
    split_row = int(len(data) * TRAIN_FRACTION)
    if split_row <= SEQUENCE_LENGTH or split_row >= len(data):
        raise ValueError("The chronological split leaves an invalid partition.")

    target_scaler = MinMaxScaler()
    multivariate_scaler = MinMaxScaler()

    target_scaler.fit(data.iloc[:split_row][[TARGET]])
    multivariate_scaler.fit(data.iloc[:split_row][MULTIVARIATE_INPUTS])

    target_scaled = target_scaler.transform(data[[TARGET]])
    multivariate_scaled = multivariate_scaler.transform(
        data[MULTIVARIATE_INPUTS]
    )

    x_univariate: list[np.ndarray] = []
    x_multivariate: list[np.ndarray] = []
    y_scaled: list[float] = []
    target_rows: list[int] = []

    for target_row in range(SEQUENCE_LENGTH, len(data)):
        start = target_row - SEQUENCE_LENGTH
        x_univariate.append(target_scaled[start:target_row])
        x_multivariate.append(multivariate_scaled[start:target_row])
        y_scaled.append(float(target_scaled[target_row, 0]))
        target_rows.append(target_row)

    x_uni = np.asarray(x_univariate, dtype=np.float32)
    x_multi = np.asarray(x_multivariate, dtype=np.float32)
    y = np.asarray(y_scaled, dtype=np.float32).reshape(-1, 1)
    target_rows_array = np.asarray(target_rows)
    train_mask = target_rows_array < split_row
    test_mask = ~train_mask

    dates = data.index.to_numpy()[target_rows_array]
    actual = data[TARGET].to_numpy()[target_rows_array]

    return {
        "X_uni_train": x_uni[train_mask],
        "X_uni_test": x_uni[test_mask],
        "X_multi_train": x_multi[train_mask],
        "X_multi_test": x_multi[test_mask],
        "y_train": y[train_mask],
        "y_test": y[test_mask],
        "actual_test": actual[test_mask],
        "dates_test": pd.DatetimeIndex(dates[test_mask]),
        "target_scaler": target_scaler,
        "multivariate_scaler": multivariate_scaler,
        "split_row": split_row,
    }


# =============================================================================
# 3. Model architectures
# =============================================================================

def build_cnn(input_shape: tuple[int, int], name: str) -> Model:
    """Build the CNN architecture used in Chapter 4."""
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(filters=32, kernel_size=5, activation="relu"),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(50, activation="relu"),
            Dense(1, activation="linear"),
        ],
        name=name,
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_lstm(input_shape: tuple[int, int], name: str) -> Model:
    """Build the LSTM architecture used in Chapter 4."""
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(50, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="linear"),
        ],
        name=name,
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_parallel_cnn_lstm(input_shape: tuple[int, int], name: str) -> Model:
    """Parallel CNN-LSTM architecture used for paired forecasting."""
    cnn_input = Input(shape=input_shape, name=f"{name}_cnn_input")
    cnn_branch = Conv1D(64, 3, activation="relu")(cnn_input)
    cnn_branch = MaxPooling1D(2)(cnn_branch)
    cnn_branch = Flatten()(cnn_branch)
    cnn_branch = Dense(50, activation="relu")(cnn_branch)

    lstm_input = Input(shape=input_shape, name=f"{name}_lstm_input")
    lstm_branch = LSTM(50, activation="relu")(lstm_input)
    lstm_branch = Dense(25, activation="relu")(lstm_branch)

    merged = concatenate([cnn_branch, lstm_branch])
    merged = Dense(50, activation="relu")(merged)
    output = Dense(1, activation="linear")(merged)

    model = Model([cnn_input, lstm_input], output, name=name)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def model_inputs(model_type: str, x: np.ndarray):
    """Return one input or the two identical hybrid-branch inputs."""
    if model_type == "CNN-LSTM":
        return [x, x]
    return x


def build_model_pair(model_type: str, feature_count: int) -> tuple[Model, Model]:
    """Build corresponding univariate and multivariate models."""
    builders = {
        "CNN": build_cnn,
        "LSTM": build_lstm,
        "CNN-LSTM": build_parallel_cnn_lstm,
    }
    builder = builders[model_type]
    safe_name = model_type.lower().replace("-", "_")
    univariate = builder(
        (SEQUENCE_LENGTH, 1),
        f"univariate_{safe_name}",
    )
    multivariate = builder(
        (SEQUENCE_LENGTH, feature_count),
        f"multivariate_{safe_name}",
    )
    return univariate, multivariate


# =============================================================================
# 4. Training, prediction, and evaluation
# =============================================================================

def training_configuration(model_type: str) -> tuple[int, int]:
    """Return the epochs and batch size for a model family."""
    configurations = {
        "CNN": (CNN_EPOCHS, CNN_BATCH_SIZE),
        "LSTM": (LSTM_EPOCHS, LSTM_BATCH_SIZE),
        "CNN-LSTM": (HYBRID_EPOCHS, HYBRID_BATCH_SIZE),
    }
    return configurations[model_type]


def train_or_load_pair(
    model_type: str,
    arrays: dict[str, object],
) -> tuple[Model, Model]:
    """Train a paired model set or load previously saved Keras models."""
    safe_name = model_type.lower().replace("-", "_")
    univariate_path = MODEL_DIR / f"univariate_{safe_name}.keras"
    multivariate_path = MODEL_DIR / f"multivariate_{safe_name}.keras"

    if not TRAIN_MODELS:
        if not univariate_path.exists() or not multivariate_path.exists():
            raise FileNotFoundError(
                f"Saved {model_type} model pair not found in {MODEL_DIR}."
            )
        return (
            tf.keras.models.load_model(univariate_path),
            tf.keras.models.load_model(multivariate_path),
        )

    tf.keras.backend.clear_session()
    set_reproducibility()
    univariate, multivariate = build_model_pair(
        model_type,
        len(MULTIVARIATE_INPUTS),
    )
    epochs, batch_size = training_configuration(model_type)

    x_uni_train = arrays["X_uni_train"]
    x_multi_train = arrays["X_multi_train"]
    y_train = arrays["y_train"]

    univariate.fit(
        model_inputs(model_type, x_uni_train),
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=VALIDATION_FRACTION,
        shuffle=False,
        verbose=1,
    )
    multivariate.fit(
        model_inputs(model_type, x_multi_train),
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=VALIDATION_FRACTION,
        shuffle=False,
        verbose=1,
    )

    univariate.save(univariate_path)
    multivariate.save(multivariate_path)
    return univariate, multivariate


def inverse_prediction(model: Model, model_type: str, x, scaler) -> np.ndarray:
    """Generate predictions and return them in the original SLV units."""
    prediction_scaled = model.predict(model_inputs(model_type, x), verbose=0)
    return scaler.inverse_transform(prediction_scaled).reshape(-1)


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate the forecasting-error metrics reported in Chapter 4."""
    mse = mean_squared_error(observed, predicted)
    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(observed, predicted)),
    }


def evaluate_pair(
    model_type: str,
    univariate: Model,
    multivariate: Model,
    arrays: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    target_scaler = arrays["target_scaler"]
    actual = np.asarray(arrays["actual_test"])
    dates = arrays["dates_test"]

    prediction_uni = inverse_prediction(
        univariate,
        model_type,
        arrays["X_uni_test"],
        target_scaler,
    )
    prediction_multi = inverse_prediction(
        multivariate,
        model_type,
        arrays["X_multi_test"],
        target_scaler,
    )

    predictions = pd.DataFrame(
        {
            "Observed_SLV": actual,
            "Univariate_prediction": prediction_uni,
            "Multivariate_prediction": prediction_multi,
        },
        index=dates,
    )
    predictions.index.name = "DATE"
    result_metrics = {
        "Univariate": metrics(actual, prediction_uni),
        "Multivariate": metrics(actual, prediction_multi),
    }
    return predictions, result_metrics


# =============================================================================
# 5. Monthly combined residual-based weights
# =============================================================================

def calculate_combined_weights(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Compare paired out-of-sample residuals by calendar month.

    Daily improvement = |univariate residual| - |multivariate residual|.
    Positive values indicate that meteorological inputs reduced error.

    The output table contains:
    1. Mean_improvement_x100: direct monthly mean difference multiplied by 100.
    2. Combined_weight_pct: each month's share of the total residual
       improvement over the testing period; these shares algebraically sum
       to 100 and may be negative when the multivariate model was worse.
    """
    frame = predictions.copy()
    frame["Univariate_residual"] = (
        frame["Observed_SLV"] - frame["Univariate_prediction"]
    )
    frame["Multivariate_residual"] = (
        frame["Observed_SLV"] - frame["Multivariate_prediction"]
    )
    frame["Daily_improvement"] = (
        frame["Univariate_residual"].abs()
        - frame["Multivariate_residual"].abs()
    )
    frame["Month_number"] = frame.index.month

    monthly = frame.groupby("Month_number")["Daily_improvement"].agg(
        Observation_count="count",
        Improvement_sum="sum",
        Improvement_mean="mean",
    )
    monthly["Mean_improvement_x100"] = monthly["Improvement_mean"] * 100.0

    total_improvement = monthly["Improvement_sum"].sum()
    if np.isclose(total_improvement, 0.0):
        monthly["Combined_weight_pct"] = np.nan
    else:
        monthly["Combined_weight_pct"] = (
            monthly["Improvement_sum"] / total_improvement * 100.0
        )

    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December",
    }
    monthly.insert(0, "Month", monthly.index.map(month_names))
    return monthly


# =============================================================================
# 6. Monthly SHAP analysis
# =============================================================================

def evenly_spaced_sample(x: np.ndarray, maximum: int) -> np.ndarray:
    """Select an evenly spaced sample with at most the requested size."""
    if len(x) <= maximum:
        return x
    indices = np.linspace(0, len(x) - 1, maximum, dtype=int)
    return x[indices]


def normalize_shap_arrays(shap_values) -> list[np.ndarray]:
    """Normalize SHAP's version-dependent one-output/multi-input structures."""
    items = shap_values if isinstance(shap_values, list) else [shap_values]
    arrays: list[np.ndarray] = []
    for item in items:
        nested = item if isinstance(item, list) else [item]
        for value in nested:
            array = np.asarray(value)
            if array.ndim == 4 and array.shape[-1] == 1:
                array = np.squeeze(array, axis=-1)
            if array.ndim == 3:
                arrays.append(array)
    if not arrays:
        raise ValueError("The returned SHAP structure could not be interpreted.")
    return arrays


def calculate_monthly_shap(
    model_type: str,
    multivariate_model: Model,
    arrays: dict[str, object],
    combined: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate monthly mean absolute SHAP values and allocate each combined
    residual-based weight proportionally among AT, AP, WS, WD, and GS.

    Historical SLV is included in the multivariate model input but is excluded
    from meteorological allocation. Signed SHAP means are saved separately to
    retain directional information without mixing sign and magnitude.
    """
    x_train = np.asarray(arrays["X_multi_train"])
    x_test_full = np.asarray(arrays["X_multi_test"])
    dates_full = pd.DatetimeIndex(arrays["dates_test"])

    if SHAP_EXPLANATION_SIZE is None or len(x_test_full) <= SHAP_EXPLANATION_SIZE:
        x_test = x_test_full
        dates = dates_full
    else:
        indices = np.linspace(
            0,
            len(x_test_full) - 1,
            SHAP_EXPLANATION_SIZE,
            dtype=int,
        )
        x_test = x_test_full[indices]
        dates = dates_full[indices]

    background = evenly_spaced_sample(x_train, SHAP_BACKGROUND_SIZE)
    if model_type == "CNN-LSTM":
        explainer = shap.GradientExplainer(
            multivariate_model,
            [background, background],
        )
        raw_values = explainer.shap_values([x_test, x_test])
    else:
        explainer = shap.GradientExplainer(multivariate_model, background)
        raw_values = explainer.shap_values(x_test)

    shap_arrays = normalize_shap_arrays(raw_values)
    # Sum the two hybrid branches; for CNN/LSTM this contains one array.
    signed = np.sum(shap_arrays, axis=0)

    # Aggregate across the 10 historical time steps for each forecast sample.
    sample_signed = signed.mean(axis=1)
    sample_absolute = np.abs(signed).mean(axis=1)
    if sample_absolute.shape[1] != len(MULTIVARIATE_INPUTS):
        raise ValueError(
            "SHAP feature count does not match the multivariate model inputs."
        )

    absolute_frame = pd.DataFrame(
        sample_absolute,
        columns=MULTIVARIATE_INPUTS,
        index=dates,
    )
    signed_frame = pd.DataFrame(
        sample_signed,
        columns=MULTIVARIATE_INPUTS,
        index=dates,
    )
    absolute_frame["Month_number"] = absolute_frame.index.month
    signed_frame["Month_number"] = signed_frame.index.month

    monthly_absolute = absolute_frame.groupby("Month_number")[
        MULTIVARIATE_INPUTS
    ].mean()
    monthly_signed = signed_frame.groupby("Month_number")[
        MULTIVARIATE_INPUTS
    ].mean()

    meteorological_absolute = monthly_absolute[METEOROLOGICAL]
    row_totals = meteorological_absolute.sum(axis=1)
    if np.isclose(row_totals, 0.0).any():
        bad_months = row_totals.index[np.isclose(row_totals, 0.0)].tolist()
        raise ZeroDivisionError(
            f"Monthly meteorological absolute SHAP total is zero: {bad_months}"
        )

    shares = meteorological_absolute.div(row_totals, axis="index")
    shares = shares.reindex(combined.index)
    allocated = shares.mul(combined["Combined_weight_pct"], axis="index")
    allocated["Combined_weight_pct"] = combined["Combined_weight_pct"]
    allocated["Allocated_sum"] = allocated[METEOROLOGICAL].sum(axis=1)
    allocated["Difference"] = (
        allocated["Allocated_sum"] - allocated["Combined_weight_pct"]
    )

    if not np.allclose(
        allocated["Difference"].dropna(),
        0.0,
        atol=1e-10,
    ):
        raise AssertionError("SHAP allocations do not equal combined weights.")

    return monthly_absolute, monthly_signed, allocated


# =============================================================================
# 7. Outputs
# =============================================================================

def save_environment(data: pd.DataFrame, arrays: dict[str, object]) -> None:
    """Save software, data, and model-configuration information."""
    information = {
        "Python": sys.version.replace("\n", " "),
        "Platform": platform.platform(),
        "TensorFlow": tf.__version__,
        "SHAP": shap.__version__,
        "pandas": pd.__version__,
        "NumPy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "Random_seed": RANDOM_SEED,
        "Sequence_length": SEQUENCE_LENGTH,
        "Train_fraction": TRAIN_FRACTION,
        "Aligned_rows": len(data),
        "Training_rows": int(arrays["split_row"]),
        "Testing_targets": len(arrays["dates_test"]),
        "Data_start": str(data.index.min()),
        "Data_end": str(data.index.max()),
    }
    pd.Series(information, name="Value").to_csv(
        OUTPUT_DIR / "chapter4_environment_and_data.csv",
        header=True,
    )

    configuration = {
        "CNN": {
            "architecture": "Conv1D(32,k=5)-MaxPool(2)-Flatten-Dense(50)-Dense(1)",
            "epochs": CNN_EPOCHS,
            "batch_size": CNN_BATCH_SIZE,
        },
        "LSTM": {
            "architecture": "LSTM(50)-Dropout(0.5)-Dense(1)",
            "epochs": LSTM_EPOCHS,
            "batch_size": LSTM_BATCH_SIZE,
        },
        "CNN-LSTM": {
            "architecture": "Parallel CNN(64,k=3,Dense50)+LSTM(50,Dense25)-Dense50-Dense1",
            "epochs": HYBRID_EPOCHS,
            "batch_size": HYBRID_BATCH_SIZE,
        },
    }
    with open(
        OUTPUT_DIR / "chapter4_model_configuration.json",
        "w",
        encoding="utf-8",
    ) as configuration_file:
        json.dump(configuration, configuration_file, indent=2)


def plot_pair(
    model_type: str,
    predictions: pd.DataFrame,
) -> None:
    """Plot observed SLV with univariate and multivariate predictions."""
    if not SAVE_PLOTS:
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(predictions.index, predictions["Observed_SLV"], label="Observed SLV")
    ax.plot(
        predictions.index,
        predictions["Univariate_prediction"],
        "--",
        label=f"Univariate {model_type}",
    )
    ax.plot(
        predictions.index,
        predictions["Multivariate_prediction"],
        label=f"Multivariate {model_type}",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Sea-level variability (m)")
    ax.legend()
    fig.tight_layout()
    safe_name = model_type.lower().replace("-", "_")
    fig.savefig(OUTPUT_DIR / f"{safe_name}_forecasts.png", dpi=300)
    plt.close(fig)


# =============================================================================
# 8. Main workflow
# =============================================================================

def main() -> None:
    """Run the complete Chapter 4 analysis."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    set_reproducibility()

    data = load_aligned_data(DATA_FILE)
    arrays = scale_and_sequence(data)
    save_environment(data, arrays)

    all_metrics: dict[str, dict[str, float]] = {}
    all_combined: list[pd.DataFrame] = []

    print(f"Aligned observations: {len(data):,}")
    print(f"Period: {data.index.min()} to {data.index.max()}")
    print(f"Testing targets: {len(arrays['dates_test']):,}")

    for model_type in ["CNN", "LSTM", "CNN-LSTM"]:
        print(f"\n{'=' * 72}\nRunning {model_type}\n{'=' * 72}")
        univariate, multivariate = train_or_load_pair(model_type, arrays)
        predictions, pair_metrics = evaluate_pair(
            model_type,
            univariate,
            multivariate,
            arrays,
        )

        safe_name = model_type.lower().replace("-", "_")
        predictions.to_csv(OUTPUT_DIR / f"{safe_name}_predictions.csv")
        plot_pair(model_type, predictions)

        for input_type, values in pair_metrics.items():
            all_metrics[f"{input_type} {model_type}"] = values

        combined = calculate_combined_weights(predictions)
        combined.to_csv(
            OUTPUT_DIR / f"{safe_name}_combined_monthly_weights.csv",
            float_format="%.8f",
        )
        combined_for_master = combined.copy()
        combined_for_master.insert(0, "Model", model_type)
        all_combined.append(combined_for_master.reset_index())

        if RUN_SHAP:
            monthly_absolute, monthly_signed, allocated = calculate_monthly_shap(
                model_type,
                multivariate,
                arrays,
                combined,
            )
            monthly_absolute.to_csv(
                OUTPUT_DIR / f"{safe_name}_monthly_mean_absolute_shap.csv",
                float_format="%.10f",
            )
            monthly_signed.to_csv(
                OUTPUT_DIR / f"{safe_name}_monthly_mean_signed_shap.csv",
                float_format="%.10f",
            )
            allocated.to_csv(
                OUTPUT_DIR / f"{safe_name}_individual_monthly_weights.csv",
                float_format="%.8f",
            )

    metrics_table = pd.DataFrame(all_metrics).T
    metrics_table.index.name = "Model"
    metrics_table.to_csv(
        OUTPUT_DIR / "chapter4_all_model_metrics.csv",
        float_format="%.8f",
    )

    pd.concat(all_combined, ignore_index=True).to_csv(
        OUTPUT_DIR / "chapter4_all_combined_monthly_weights.csv",
        index=False,
        float_format="%.8f",
    )

    print("\nChapter 4 model metrics\n")
    print(metrics_table)
    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
