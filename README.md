# DataDrivenClimateForecasting

Climate Change Trend Analyzer using real datasets from Kaggle or NASA.

## What this project does

- Loads historical temperature anomaly data from CSV.
- Cleans and aggregates data by year.
- Visualizes historical climate trend.
- Compares linear vs quadratic forecasting models with backtesting.
- Adds confidence intervals to future predictions.
- Downloads real climate data with a built-in fetch script.
- Provides an interactive Streamlit frontend with Plotly charts.

## Project structure

```text
DataDrivenClimateForecasting/
  data/
    raw/
      GlobalTemperatures.csv
  outputs/
  app.py
  src/
    climate_trend_analyzer.py
    fetch_climate_data.py
  requirements.txt
```

## Quick start

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download a real dataset (Kaggle default):

```bash
python -m src.fetch_climate_data --source kaggle
```

4. Run analysis:

```bash
python -m src.climate_trend_analyzer
```

5. Launch interactive frontend:

```bash
streamlit run app.py
```

This creates:
- `outputs/trend_forecast.png`
- `outputs/forecast.csv`
- `outputs/model_metrics.csv`
- `outputs/analysis_summary.txt`

## Dataset options

Kaggle (default):

```bash
python -m src.fetch_climate_data --source kaggle
```

NASA GISS (alternative):

```bash
python -m src.fetch_climate_data --source nasa
```

If your own dataset has explicit year/temp-anomaly columns, pass them with flags:

```bash
python -m src.climate_trend_analyzer \
  --input data/raw/your_dataset.csv \
  --year-col Year \
  --temp-col Temperature_Anomaly_C \
  --forecast-years 30 \
  --model auto \
  --test-years 10 \
  --confidence 0.95
```

## Suggested free datasets

- Kaggle: Berkeley Earth climate dataset (`GlobalTemperatures.csv`)
- NASA GISS: `GLB.Ts+dSST.csv`

## Notes

- `--model auto` picks the best backtested model by RMSE.
- Supported confidence levels: `0.8`, `0.9`, `0.95`, `0.99`.
- For stronger research output, consider ARIMA/Prophet/LSTM with probabilistic forecasting.
- Kaggle format (`dt`, `LandAverageTemperature`) is handled automatically.
- Streamlit app includes sidebar controls, interactive Plotly chart, and CSV downloads.
