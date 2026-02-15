from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

# Keep Matplotlib cache inside project to avoid home-directory permission issues.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze historical climate trends and forecast future temperature anomaly."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/GlobalTemperatures.csv"),
        help="Path to real climate CSV (for example Kaggle GlobalTemperatures.csv).",
    )
    parser.add_argument(
        "--year-col",
        type=str,
        default="Year",
        help="Column name for year values.",
    )
    parser.add_argument(
        "--temp-col",
        type=str,
        default="Temperature_Anomaly_C",
        help="Column name for temperature anomaly values in Celsius.",
    )
    parser.add_argument(
        "--forecast-years",
        type=int,
        default=20,
        help="Number of years to forecast into the future.",
    )
    parser.add_argument(
        "--model",
        choices=["auto", "linear", "quadratic"],
        default="auto",
        help="Forecast model choice. 'auto' picks best model by test RMSE.",
    )
    parser.add_argument(
        "--test-years",
        type=int,
        default=10,
        help="Number of most recent years reserved for backtesting.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for forecast interval (0.8, 0.9, 0.95, 0.99).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where chart and forecast CSV are written.",
    )
    return parser.parse_args()


def load_and_prepare_data(input_path: Path, year_col: str, temp_col: str) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}. Download a real dataset first."
        )

    df = pd.read_csv(input_path, na_values=["***"])
    if len(df.columns) == 1 and "Land-Ocean: Global Means" in str(df.columns[0]):
        # NASA GISS file includes a title row before the actual header.
        df = pd.read_csv(input_path, skiprows=1, na_values=["***"])
    if year_col in df.columns and temp_col in df.columns:
        cleaned = (
            df[[year_col, temp_col]]
            .rename(columns={year_col: "year", temp_col: "temp_anomaly_c"})
            .dropna()
        )
        cleaned["year"] = cleaned["year"].astype(int)
        cleaned["temp_anomaly_c"] = pd.to_numeric(cleaned["temp_anomaly_c"], errors="coerce")
        cleaned = cleaned.dropna()
        return (
            cleaned.groupby("year", as_index=False)["temp_anomaly_c"]
            .mean()
            .sort_values("year")
            .reset_index(drop=True)
        )

    # Kaggle Berkeley Earth format fallback.
    if {"dt", "LandAverageTemperature"}.issubset(df.columns):
        kaggle = df[["dt", "LandAverageTemperature"]].copy()
        kaggle["year"] = pd.to_datetime(kaggle["dt"], errors="coerce").dt.year
        kaggle["temp_c"] = pd.to_numeric(
            kaggle["LandAverageTemperature"], errors="coerce"
        )
        kaggle = kaggle.dropna(subset=["year", "temp_c"])
        yearly_mean = (
            kaggle.groupby("year", as_index=False)["temp_c"]
            .mean()
            .sort_values("year")
            .reset_index(drop=True)
        )
        baseline = yearly_mean["temp_c"].mean()
        yearly_mean["temp_anomaly_c"] = yearly_mean["temp_c"] - baseline
        return yearly_mean[["year", "temp_anomaly_c"]]

    # NASA GISS table format fallback (annual anomaly in J-D column).
    if {"Year", "J-D"}.issubset(df.columns):
        nasa = df[["Year", "J-D"]].copy()
        nasa = nasa.rename(columns={"Year": "year", "J-D": "temp_anomaly_c"})
        nasa["year"] = pd.to_numeric(nasa["year"], errors="coerce")
        nasa["temp_anomaly_c"] = pd.to_numeric(nasa["temp_anomaly_c"], errors="coerce")
        nasa = nasa.dropna(subset=["year", "temp_anomaly_c"])
        nasa["year"] = nasa["year"].astype(int)
        return nasa.sort_values("year").reset_index(drop=True)

    raise ValueError(
        "Unsupported dataset schema. Provide either "
        f"('{year_col}', '{temp_col}') columns, "
        "Kaggle columns ('dt', 'LandAverageTemperature'), "
        "or NASA GISS columns ('Year', 'J-D')."
    )


def train_test_split_time_series(yearly: pd.DataFrame, test_years: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_rows = len(yearly)
    if n_rows < 8:
        raise ValueError("Need at least 8 yearly records for robust backtesting.")
    test_years = max(3, min(test_years, n_rows // 3))
    split_idx = n_rows - test_years
    train = yearly.iloc[:split_idx].reset_index(drop=True)
    test = yearly.iloc[split_idx:].reset_index(drop=True)
    return train, test


def fit_poly_model(train_df: pd.DataFrame, degree: int) -> np.ndarray:
    x_train = train_df["year"].to_numpy()
    y_train = train_df["temp_anomaly_c"].to_numpy()
    return np.polyfit(x_train, y_train, degree)


def predict_poly(coeffs: np.ndarray, years: np.ndarray) -> np.ndarray:
    poly = np.poly1d(coeffs)
    return poly(years)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return {"mae": mae, "rmse": rmse}


def evaluate_models(
    train_df: pd.DataFrame, test_df: pd.DataFrame, user_model: str
) -> tuple[str, np.ndarray, pd.DataFrame]:
    candidate_degrees: list[tuple[str, int]]
    if user_model == "linear":
        candidate_degrees = [("linear", 1)]
    elif user_model == "quadratic":
        candidate_degrees = [("quadratic", 2)]
    else:
        candidate_degrees = [("linear", 1), ("quadratic", 2)]

    test_years = test_df["year"].to_numpy()
    y_test = test_df["temp_anomaly_c"].to_numpy()

    rows: list[dict[str, float | str]] = []
    best_model_name = ""
    best_coeffs = np.array([])
    best_rmse = math.inf

    for model_name, degree in candidate_degrees:
        coeffs = fit_poly_model(train_df, degree)
        preds = predict_poly(coeffs, test_years)
        metrics = calc_metrics(y_test, preds)
        rows.append({"model": model_name, "mae": metrics["mae"], "rmse": metrics["rmse"]})
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_model_name = model_name
            best_coeffs = coeffs

    return best_model_name, best_coeffs, pd.DataFrame(rows)


def z_score_for_confidence(confidence: float) -> float:
    z_map = {0.8: 1.282, 0.9: 1.645, 0.95: 1.96, 0.99: 2.576}
    rounded = round(confidence, 2)
    if rounded not in z_map:
        raise ValueError("Supported confidence levels: 0.8, 0.9, 0.95, 0.99")
    return z_map[rounded]


def build_forecast(
    yearly: pd.DataFrame, coeffs: np.ndarray, forecast_years: int, z_score: float
) -> pd.DataFrame:
    residuals = yearly["temp_anomaly_c"].to_numpy() - predict_poly(
        coeffs, yearly["year"].to_numpy()
    )
    residual_std = float(np.std(residuals, ddof=1))

    last_year = int(yearly["year"].max())
    future_years = np.arange(last_year + 1, last_year + forecast_years + 1)
    future_temps = predict_poly(coeffs, future_years)
    half_width = z_score * residual_std

    forecast = pd.DataFrame(
        {
            "year": future_years,
            "predicted_temp_anomaly_c": future_temps,
            "lower_bound_c": future_temps - half_width,
            "upper_bound_c": future_temps + half_width,
        }
    )
    return forecast


def plot_trend_and_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast: pd.DataFrame,
    model_name: str,
    confidence: float,
    output_path: Path,
) -> None:
    plt.figure(figsize=(11, 6))
    plt.plot(
        train_df["year"], train_df["temp_anomaly_c"], label="Train (Historical)", linewidth=2
    )
    plt.plot(test_df["year"], test_df["temp_anomaly_c"], label="Test (Backtest)", linewidth=2)
    plt.plot(
        forecast["year"],
        forecast["predicted_temp_anomaly_c"],
        "--",
        label=f"Forecast ({model_name})",
        linewidth=2,
    )
    plt.fill_between(
        forecast["year"],
        forecast["lower_bound_c"],
        forecast["upper_bound_c"],
        alpha=0.2,
        label=f"{int(confidence * 100)}% Confidence Band",
    )
    plt.title("Global Temperature Anomaly Forecast with Backtesting")
    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly (°C)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    yearly = load_and_prepare_data(args.input, args.year_col, args.temp_col)
    train_df, test_df = train_test_split_time_series(yearly, args.test_years)
    best_model_name, best_coeffs, metrics_df = evaluate_models(train_df, test_df, args.model)
    z_score = z_score_for_confidence(args.confidence)
    forecast = build_forecast(yearly, best_coeffs, args.forecast_years, z_score)

    forecast_csv_path = args.output_dir / "forecast.csv"
    metrics_csv_path = args.output_dir / "model_metrics.csv"
    plot_path = args.output_dir / "trend_forecast.png"
    summary_path = args.output_dir / "analysis_summary.txt"

    forecast.to_csv(forecast_csv_path, index=False)
    metrics_df.to_csv(metrics_csv_path, index=False)
    plot_trend_and_forecast(
        train_df, test_df, forecast, best_model_name, args.confidence, plot_path
    )
    best_metrics = (
        metrics_df.loc[metrics_df["model"] == best_model_name].to_dict(orient="records")[0]
    )
    summary_lines = [
        "Climate trend analysis summary",
        f"Input file: {args.input}",
        f"Rows used (yearly): {len(yearly)}",
        f"Train years: {len(train_df)}",
        f"Test years: {len(test_df)}",
        f"Selected model: {best_model_name}",
        f"Backtest MAE: {best_metrics['mae']:.4f}",
        f"Backtest RMSE: {best_metrics['rmse']:.4f}",
        f"Forecast horizon: {args.forecast_years} years",
        f"Confidence level: {args.confidence}",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("Climate trend analysis complete.")
    print(f"Selected model: {best_model_name}")
    print(f"Forecast CSV: {forecast_csv_path}")
    print(f"Model metrics CSV: {metrics_csv_path}")
    print(f"Trend plot: {plot_path}")
    print(f"Summary report: {summary_path}")


if __name__ == "__main__":
    main()
