from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.climate_trend_analyzer import (
    build_forecast,
    evaluate_models,
    load_and_prepare_data,
    predict_poly,
    train_test_split_time_series,
    z_score_for_confidence,
)
from src.fetch_climate_data import fetch_from_kaggle, fetch_from_nasa


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def render_theme() -> None:
    st.set_page_config(
        page_title="Climate Change Trend Analyzer",
        page_icon="🌍",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 15% 20%, #d1f7ff 0%, #f7fbff 35%, #f3f8e8 100%);
            color: #0f172a;
        }
        .hero {
            background: linear-gradient(120deg, #0b7285 0%, #2f9e44 100%);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            color: white;
            box-shadow: 0 10px 24px rgba(0,0,0,0.12);
            margin-bottom: 1rem;
        }
        .metric-box {
            background: rgba(255, 255, 255, 0.78);
            border-radius: 14px;
            padding: 0.8rem 1rem;
            border: 1px solid rgba(12, 74, 110, 0.16);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fetch_dataset_ui(data_dir: Path) -> None:
    st.sidebar.subheader("Download Dataset")
    source = st.sidebar.selectbox("Source", ["kaggle", "nasa"], index=0)
    if source == "kaggle":
        dataset_slug = st.sidebar.text_input(
            "Kaggle dataset slug",
            value="berkeleyearth/climate-change-earth-surface-temperature-data",
        )
        filename = st.sidebar.text_input("File to extract", value="GlobalTemperatures.csv")
    else:
        dataset_slug = ""
        filename = ""

    if st.sidebar.button("Fetch dataset", use_container_width=True):
        try:
            if source == "kaggle":
                saved_path = fetch_from_kaggle(data_dir, dataset_slug, filename)
            else:
                saved_path = fetch_from_nasa(data_dir)
            st.session_state["input_csv_path"] = str(saved_path)
            st.sidebar.success(f"Saved: {saved_path}")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(f"Fetch failed: {exc}")


def main() -> None:
    render_theme()
    st.markdown(
        """
        <div class="hero">
          <h2 style="margin:0;">Climate Change Trend Analyzer</h2>
          <p style="margin:0.25rem 0 0 0;">
            Interactive forecasting with backtesting, model comparison, and uncertainty bands.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    fetch_dataset_ui(data_dir)

    st.sidebar.subheader("Analysis Settings")
    default_input = data_dir / "GlobalTemperatures.csv"
    if "input_csv_path" not in st.session_state:
        st.session_state["input_csv_path"] = str(default_input)
    input_path_text = st.sidebar.text_input("Input CSV path", key="input_csv_path")
    input_path = Path(input_path_text)

    uploaded_file = st.sidebar.file_uploader("Or upload a CSV", type=["csv"])
    if uploaded_file is not None:
        upload_path = data_dir / uploaded_file.name
        upload_path.write_bytes(uploaded_file.getvalue())
        st.session_state["input_csv_path"] = str(upload_path)
        input_path = upload_path
        st.sidebar.info(f"Uploaded to: {upload_path}")

    year_col = st.sidebar.text_input("Year column", value="Year")
    temp_col = st.sidebar.text_input("Temperature column", value="Temperature_Anomaly_C")
    model = st.sidebar.selectbox("Model", ["auto", "linear", "quadratic"], index=0)
    forecast_years = st.sidebar.slider("Forecast years", min_value=5, max_value=100, value=30, step=5)
    test_years = st.sidebar.slider("Backtest years", min_value=3, max_value=40, value=10, step=1)
    confidence = st.sidebar.select_slider("Confidence", options=[0.8, 0.9, 0.95, 0.99], value=0.95)

    run_clicked = st.sidebar.button("Run analysis", type="primary", use_container_width=True)
    if not run_clicked:
        st.info("Configure inputs on the left and click `Run analysis`.")
        return

    try:
        yearly = load_and_prepare_data(input_path, year_col, temp_col)
        train_df, test_df = train_test_split_time_series(yearly, test_years)
        best_model_name, best_coeffs, metrics_df = evaluate_models(train_df, test_df, model)
        z_score = z_score_for_confidence(confidence)
        forecast_df = build_forecast(yearly, best_coeffs, forecast_years, z_score)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Analysis failed: {exc}")
        return

    test_pred_df = pd.DataFrame(
        {
            "year": test_df["year"],
            "predicted_temp_anomaly_c": predict_poly(best_coeffs, test_df["year"].to_numpy()),
        }
    )
    best_row = metrics_df.loc[metrics_df["model"] == best_model_name].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        f'<div class="metric-box"><b>Selected Model</b><br>{best_model_name}</div>',
        unsafe_allow_html=True,
    )
    col2.markdown(
        f'<div class="metric-box"><b>Backtest MAE</b><br>{best_row["mae"]:.4f}</div>',
        unsafe_allow_html=True,
    )
    col3.markdown(
        f'<div class="metric-box"><b>Backtest RMSE</b><br>{best_row["rmse"]:.4f}</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_df["year"],
            y=train_df["temp_anomaly_c"],
            mode="lines",
            name="Train (Historical)",
            line={"width": 3, "color": "#0b7285"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_df["year"],
            y=test_df["temp_anomaly_c"],
            mode="lines+markers",
            name="Test (Actual)",
            line={"width": 3, "color": "#e67700"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_pred_df["year"],
            y=test_pred_df["predicted_temp_anomaly_c"],
            mode="lines",
            name="Test (Predicted)",
            line={"width": 2, "dash": "dot", "color": "#495057"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["year"],
            y=forecast_df["predicted_temp_anomaly_c"],
            mode="lines",
            name="Forecast",
            line={"width": 3, "dash": "dash", "color": "#2f9e44"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["year"],
            y=forecast_df["upper_bound_c"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["year"],
            y=forecast_df["lower_bound_c"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(47, 158, 68, 0.2)",
            line={"width": 0},
            name=f"{int(confidence * 100)}% Confidence Band",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title="Global Temperature Anomaly: Backtesting + Forecast",
        xaxis_title="Year",
        yaxis_title="Temperature Anomaly (C)",
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 30, "r": 30, "t": 70, "b": 30},
    )
    st.plotly_chart(fig, use_container_width=True)

    metrics_col, forecast_col = st.columns(2)
    with metrics_col:
        st.subheader("Model Metrics")
        st.dataframe(metrics_df, use_container_width=True)
        st.download_button(
            "Download metrics CSV",
            data=to_csv_bytes(metrics_df),
            file_name="model_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with forecast_col:
        st.subheader("Forecast Table")
        st.dataframe(forecast_df, use_container_width=True)
        st.download_button(
            "Download forecast CSV",
            data=to_csv_bytes(forecast_df),
            file_name="forecast.csv",
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
