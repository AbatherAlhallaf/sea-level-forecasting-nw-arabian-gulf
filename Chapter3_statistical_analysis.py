"""
Chapter 3: Statistical Analysis of Sea-Level Variability

This script implements the statistical analyses presented in Chapter 3:

1. Prepare daily sea-level variability (SLV) and meteorological data.
2. Compare ARIMA and SARIMA using the univariate SLV dataset.
3. Compare SARIMA and SARIMAX using the aligned multivariate dataset.
4. Calculate forecasting-error metrics and residual diagnostics.
5. Estimate model-based monthly meteorological contributions.

Required input files
--------------------
Place the following files in a folder named "data" beside this script:

data/Cleaned_univariate_data.csv
    Required columns: DATE and SLV (or WL1).

data/Cleaned_SARIMAX_data.csv
    Required columns: DATE, SLV (or WL1), AT (or AT2), AP (or P),
    WS, WD (or GD), and GS.

Results are saved in the "chapter3_outputs" folder.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import statsmodels
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


# ---------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "chapter3_outputs"

UNIVARIATE_FILE = DATA_DIR / "Cleaned_univariate_data.csv"
ALIGNED_FILE = DATA_DIR / "Cleaned_SARIMAX_data.csv"

DATE_COLUMN = "DATE"
TARGET = "SLV"
EXOGENOUS = ["AT", "AP", "WS", "WD", "GS"]
TRAIN_FRACTION = 0.80
SEASONAL_PERIOD = 12

# Analysis 1 model configurations
ARIMA_ORDER = (1, 1, 1)
SARIMA1_ORDER = (3, 1, 1)
SARIMA1_SEASONAL_ORDER = (1, 1, 0, SEASONAL_PERIOD)

# Analysis 2 model configurations
SARIMA2_ORDER = (3, 1, 1)
SARIMA2_SEASONAL_ORDER = (3, 1, 0, SEASONAL_PERIOD)
SARIMAX_ORDER = (3, 1, 1)
SARIMAX_SEASONAL_ORDER = (3, 1, 0, SEASONAL_PERIOD)

# ---------------------------------------------------------------------
# 2. Reusable functions
# ---------------------------------------------------------------------

def canonicalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Convert historical column names to the dissertation terminology."""
    aliases = {
        "WL1": "SLV",
        "AT2": "AT",
        "P": "AP",
        "GD": "WD",
    }
    rename_map = {
        old: new
        for old, new in aliases.items()
        if old in data.columns and new not in data.columns
    }
    return data.rename(columns=rename_map)


def load_daily_data(path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Load, validate, and convert the observations to daily values."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Place the final CSV file in the data folder beside this script."
        )

    data = pd.read_csv(path)
    data = canonicalize_columns(data)

    if DATE_COLUMN not in data.columns:
        raise ValueError(f"{path.name} does not contain a {DATE_COLUMN} column.")

    missing_columns = [c for c in required_columns if c not in data.columns]
    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required columns: {missing_columns}"
        )

    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    data = data.dropna(subset=[DATE_COLUMN]).sort_values(DATE_COLUMN)
    data = data.drop_duplicates(subset=[DATE_COLUMN], keep="first")

    for column in required_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Convert the historical missing-value indicator to NaN.
    data[required_columns] = data[required_columns].replace(-999, np.nan)

    data = data.set_index(DATE_COLUMN)

    # This works for both 10-minute and already-daily files. Each daily value
    # is the arithmetic mean of the valid observations recorded that day.
    daily = data[required_columns].resample("D").mean()

    # A univariate file drops dates missing SLV only. An aligned multivariate
    # file drops dates missing any required variable. No interpolation is used.
    daily = daily.dropna(subset=required_columns)
    return daily


def chronological_split(
    data: pd.DataFrame,
    fraction: float = TRAIN_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a non-random chronological training/testing split."""
    split_index = int(len(data) * fraction)
    if split_index <= 0 or split_index >= len(data):
        raise ValueError("The requested split leaves an empty partition.")
    return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()


def forecast_metrics(observed: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Return the three forecasting metrics reported in Chapter 3."""
    observed, predicted = observed.align(predicted, join="inner")
    mse = mean_squared_error(observed, predicted)
    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(observed, predicted)),
    }


def fit_model(
    endog: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    exog: pd.DataFrame | None = None,
):
    """Fit an ARIMA-family model with reproducible optimizer settings."""
    model = SARIMAX(
        endog=endog,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=1000)


def save_metrics(metrics: dict[str, dict[str, float]], filename: str) -> None:
    table = pd.DataFrame(metrics).T
    table.index.name = "Model"
    table.to_csv(OUTPUT_DIR / filename, float_format="%.8f")
    print(f"\n{filename}\n{table}")


def save_ljung_box(result, model_name: str, filename: str) -> None:
    """Evaluate residual autocorrelation at valid seasonal lags."""
    residuals = pd.Series(result.resid).dropna()
    candidate_lags = [SEASONAL_PERIOD, 2 * SEASONAL_PERIOD]
    valid_lags = [lag for lag in candidate_lags if lag < len(residuals) / 5]
    if not valid_lags:
        valid_lags = [min(10, max(1, len(residuals) // 5))]
    diagnostic = acorr_ljungbox(residuals, lags=valid_lags, return_df=True)
    diagnostic.insert(0, "Model", model_name)
    diagnostic.to_csv(OUTPUT_DIR / filename, index_label="Lag")


def save_environment() -> None:
    """Record software versions needed for reproducibility."""
    versions = pd.Series(
        {
            "Python": sys.version.replace("\n", " "),
            "Platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "matplotlib": plt.matplotlib.__version__,
            "statsmodels": statsmodels.__version__,
            "scikit-learn": sklearn.__version__,
        },
        name="Version",
    )
    versions.to_csv(OUTPUT_DIR / "software_versions.csv", header=True)


# ---------------------------------------------------------------------
# 3. Analysis 1: ARIMA and SARIMA comparison
# ---------------------------------------------------------------------

def run_analysis_1(univariate: pd.DataFrame) -> None:
    train, test = chronological_split(univariate[[TARGET]])

    adf = adfuller(train[TARGET].dropna())
    pd.Series(
        {
            "ADF statistic": adf[0],
            "p-value": adf[1],
            "used lags": adf[2],
            "observations": adf[3],
        }
    ).to_csv(OUTPUT_DIR / "analysis1_adf_test.csv", header=False)

    arima_result = fit_model(train[TARGET], ARIMA_ORDER)
    sarima_result = fit_model(
        train[TARGET], SARIMA1_ORDER, SARIMA1_SEASONAL_ORDER
    )

    arima_prediction = arima_result.get_forecast(len(test)).predicted_mean
    sarima_prediction = sarima_result.get_forecast(len(test)).predicted_mean
    arima_prediction.index = test.index
    sarima_prediction.index = test.index

    metrics = {
        "ARIMA": forecast_metrics(test[TARGET], arima_prediction),
        "SARIMA": forecast_metrics(test[TARGET], sarima_prediction),
    }
    save_metrics(metrics, "analysis1_arima_sarima_metrics.csv")

    save_ljung_box(arima_result, "ARIMA", "analysis1_arima_ljung_box.csv")
    save_ljung_box(sarima_result, "SARIMA", "analysis1_sarima_ljung_box.csv")

    predictions = pd.DataFrame(
        {
            "Observed_SLV": test[TARGET],
            "ARIMA": arima_prediction,
            "SARIMA": sarima_prediction,
        }
    )
    predictions.to_csv(OUTPUT_DIR / "analysis1_predictions.csv")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test.index, test[TARGET], color="tab:blue", label="Observed SLV")
    ax.plot(test.index, arima_prediction, "--", color="green", label="ARIMA")
    ax.plot(test.index, sarima_prediction, "--", color="red", label="SARIMA")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sea-level variability (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "analysis1_arima_sarima_forecasts.png", dpi=300)
    plt.close(fig)

    with open(OUTPUT_DIR / "analysis1_arima_summary.txt", "w", encoding="utf-8") as f:
        f.write(arima_result.summary().as_text())
    with open(OUTPUT_DIR / "analysis1_sarima_summary.txt", "w", encoding="utf-8") as f:
        f.write(sarima_result.summary().as_text())


# ---------------------------------------------------------------------
# 4. Analysis 2: SARIMA and SARIMAX comparison
# ---------------------------------------------------------------------

def standardize_exogenous(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Fit z-score standardization on the training exogenous variables and
    apply the same transformation to the testing observations.

    For variable i:
        X*_i,t = (X_i,t - training_mean_i) / training_std_i

    Fitting the scaler on training observations prevents information from
    the testing period from entering model development.
    """
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train[EXOGENOUS]),
        columns=EXOGENOUS,
        index=train.index,
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test[EXOGENOUS]),
        columns=EXOGENOUS,
        index=test.index,
    )
    return train_scaled, test_scaled, scaler


def calculate_monthly_weights(
    test: pd.DataFrame,
    sarima_prediction: pd.Series,
    sarimax_prediction: pd.Series,
    standardized_exog_test: pd.DataFrame,
    sarimax_result,
) -> pd.DataFrame:
    """
    Calculate the residual-based monthly weights reported in Chapter 3.

    Combined monthly weight:
        T_m = mean(|e_SARIMA| - |e_SARIMAX|) * 100

    Raw score for factor i in month m:
        R_i,m = beta_i * mean(X*_i,t in month m)

    Allocated individual weight:
        S_i,m = T_m * R_i,m / sum_i(R_i,m)

    The signs of the SARIMAX coefficients are retained, and the algebraic
    sum of the five individual weights equals the combined monthly weight.
    """
    residual_frame = pd.DataFrame(
        {
            "SARIMA": test[TARGET] - sarima_prediction,
            "SARIMAX": test[TARGET] - sarimax_prediction,
        },
        index=test.index,
    )
    residual_frame["MONTH"] = residual_frame.index.month
    residual_frame["RESIDUAL_IMPROVEMENT"] = (
        residual_frame["SARIMA"].abs()
        - residual_frame["SARIMAX"].abs()
    )

    combined = (
        residual_frame.groupby("MONTH")["RESIDUAL_IMPROVEMENT"].mean()
        * 100.0
    )

    monthly_exog = standardized_exog_test.copy()
    monthly_exog["MONTH"] = monthly_exog.index.month
    monthly_means = monthly_exog.groupby("MONTH")[EXOGENOUS].mean()

    coefficients = pd.Series(
        {variable: float(sarimax_result.params[variable]) for variable in EXOGENOUS}
    )
    raw_scores = monthly_means.mul(coefficients, axis="columns")
    denominators = raw_scores.sum(axis=1)

    near_zero = denominators.abs() < np.finfo(float).eps
    if near_zero.any():
        bad_months = list(denominators.index[near_zero])
        raise ZeroDivisionError(
            f"Raw meteorological scores sum to zero for months {bad_months}."
        )

    allocated = raw_scores.div(denominators, axis="index").mul(combined, axis="index")
    allocated["Combined_weight"] = combined
    allocated.index.name = "Month_number"

    # Reproducibility assertion: individual algebraic sum must equal total.
    difference = allocated[EXOGENOUS].sum(axis=1) - allocated["Combined_weight"]
    if not np.allclose(difference, 0.0, atol=1e-10):
        raise AssertionError("Individual weights do not sum to the combined weight.")

    month_names = pd.Series(
        {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December",
        }
    )
    allocated.insert(0, "Month", allocated.index.map(month_names))
    return allocated


def run_analysis_2(aligned: pd.DataFrame) -> None:
    train, test = chronological_split(aligned[[TARGET] + EXOGENOUS])
    exog_train, exog_test, scaler = standardize_exogenous(train, test)

    sarima_result = fit_model(
        train[TARGET], SARIMA2_ORDER, SARIMA2_SEASONAL_ORDER
    )
    sarimax_result = fit_model(
        train[TARGET],
        SARIMAX_ORDER,
        SARIMAX_SEASONAL_ORDER,
        exog=exog_train,
    )

    sarima_prediction = sarima_result.get_forecast(len(test)).predicted_mean
    sarimax_prediction = sarimax_result.get_forecast(
        len(test), exog=exog_test
    ).predicted_mean
    sarima_prediction.index = test.index
    sarimax_prediction.index = test.index

    metrics = {
        "SARIMA": forecast_metrics(test[TARGET], sarima_prediction),
        "SARIMAX": forecast_metrics(test[TARGET], sarimax_prediction),
    }
    save_metrics(metrics, "analysis2_sarima_sarimax_metrics.csv")

    predictions = pd.DataFrame(
        {
            "Observed_SLV": test[TARGET],
            "SARIMA": sarima_prediction,
            "SARIMAX": sarimax_prediction,
        }
    )
    predictions.to_csv(OUTPUT_DIR / "analysis2_predictions.csv")

    coefficients = sarimax_result.params.reindex(EXOGENOUS)
    coefficients.to_csv(
        OUTPUT_DIR / "analysis2_sarimax_exogenous_coefficients.csv",
        header=["Coefficient"],
    )

    scaler_table = pd.DataFrame(
        {
            "Variable": EXOGENOUS,
            "Training_mean": scaler.mean_,
            "Training_standard_deviation": scaler.scale_,
            "Training_variance": scaler.var_,
        }
    )
    scaler_table.to_csv(
        OUTPUT_DIR / "analysis2_standardization_parameters.csv",
        index=False,
    )

    weights = calculate_monthly_weights(
        test=test,
        sarima_prediction=sarima_prediction,
        sarimax_prediction=sarimax_prediction,
        standardized_exog_test=exog_test,
        sarimax_result=sarimax_result,
    )
    weights.to_csv(
        OUTPUT_DIR / "analysis2_monthly_meteorological_weights.csv",
        float_format="%.6f",
    )
    print("\nMonthly meteorological weights\n", weights)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test.index, test[TARGET], color="tab:blue", label="Observed SLV")
    ax.plot(test.index, sarima_prediction, "--", color="green", label="SARIMA")
    ax.plot(test.index, sarimax_prediction, color="red", label="SARIMAX")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sea-level variability (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "analysis2_sarima_sarimax_forecasts.png", dpi=300)
    plt.close(fig)

    with open(OUTPUT_DIR / "analysis2_sarima_summary.txt", "w", encoding="utf-8") as f:
        f.write(sarima_result.summary().as_text())
    with open(OUTPUT_DIR / "analysis2_sarimax_summary.txt", "w", encoding="utf-8") as f:
        f.write(sarimax_result.summary().as_text())


# ---------------------------------------------------------------------
# 5. Run the complete workflow
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_environment()

    univariate = load_daily_data(UNIVARIATE_FILE, [TARGET])
    aligned = load_daily_data(ALIGNED_FILE, [TARGET] + EXOGENOUS)

    print(f"Univariate observations: {len(univariate):,}")
    print(f"Aligned multivariate observations: {len(aligned):,}")
    print(f"Univariate period: {univariate.index.min()} to {univariate.index.max()}")
    print(f"Aligned period: {aligned.index.min()} to {aligned.index.max()}")

    run_analysis_1(univariate)
    run_analysis_2(aligned)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
