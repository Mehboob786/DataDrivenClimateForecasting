# DataDrivenClimateForecasting

Interactive climate trend analysis and forecasting using real datasets from Kaggle or NASA GISS.

## Features

- Loads and standardizes multiple dataset schemas.
- Supports:
  - Kaggle Berkeley Earth format (`dt`, `LandAverageTemperature`)
  - NASA GISS format (`Year`, `J-D`) including title-row files (`Land-Ocean: Global Means`)
  - Custom files with explicit year and anomaly columns
- Forecasting models: `linear`, `quadratic`, `auto` (best RMSE on backtest).
- Time-series backtesting with MAE and RMSE.
- Forecast confidence bands (`0.8`, `0.9`, `0.95`, `0.99`).
- Interactive Streamlit + Plotly frontend with CSV download buttons.

## Project Structure

```text
DataDrivenClimateForecasting/
  app.py
  data/
    raw/
      GlobalTemperatures.csv
      GLB.Ts+dSST.csv
  outputs/
    forecast.csv
    model_metrics.csv
    trend_forecast.png
    analysis_summary.txt
  src/
    climate_trend_analyzer.py
    fetch_climate_data.py
  requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Dataset

Kaggle (default):

```bash
python -m src.fetch_climate_data --source kaggle
```

NASA GISS:

```bash
python -m src.fetch_climate_data --source nasa
```

## Run CLI Analysis

Default run:

```bash
python -m src.climate_trend_analyzer
```

Custom run:

```bash
python -m src.climate_trend_analyzer \
  --input data/raw/GLB.Ts+dSST.csv \
  --model auto \
  --forecast-years 30 \
  --test-years 10 \
  --confidence 0.95
```

Generated outputs:

- `outputs/forecast.csv`
- `outputs/model_metrics.csv`
- `outputs/trend_forecast.png`
- `outputs/analysis_summary.txt`

## Run Interactive Frontend

```bash
streamlit run app.py
```

In the app:

- Use the sidebar to fetch Kaggle/NASA datasets.
- After fetch, input path auto-updates to the downloaded file.
- Uploading a CSV also auto-updates the input path.
- Click `Run analysis` to render charts and tables.

## CLI Options

- `--input` path to CSV file
- `--year-col` custom year column (for custom schema)
- `--temp-col` custom anomaly column (for custom schema)
- `--model` `auto|linear|quadratic`
- `--forecast-years` forecast horizon
- `--test-years` backtest window size
- `--confidence` `0.8|0.9|0.95|0.99`
- `--output-dir` output folder (default `outputs`)

## Notes

- `--model auto` selects the best model by backtest RMSE.
- Kaggle CLI must be configured to fetch Kaggle datasets.
- NASA files may include `***` placeholders; these are treated as missing values.
