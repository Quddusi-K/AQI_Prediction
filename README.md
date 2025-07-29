# AQI Multi-City Forecasting & Analysis Dashboard

A robust, interactive dashboard for 72-hour Air Quality Index (AQI) forecasting, model explainability, and exploratory data analysis (EDA) for Karachi, Islamabad, and Lahore. The project features automated data fetching, multi-model predictions, SHAP explainability, and a modern Streamlit UI with downloadable visualizations.

---

## Features

- **Multi-City Support:** Forecasts and analysis for Karachi, Islamabad, and Lahore.
- **Automated Data Pipeline:**
  - Fetches and appends latest air quality and weather data from Open-Meteo APIs.
  - Handles time overlap and prioritizes recent data.
  - CI/CD workflows for daily data and forecast updates.
- **Model Predictions:**
  - Uses Ridge Regression and other models (Random Forest, Gradient Boosting, Linear Regression) for AQI prediction.
  - Saves all features and predictions for transparency.
- **Model Explainability:**
  - SHAP analysis for each city, with downloadable summary plots.
- **Interactive Streamlit Dashboard:**
  - City selection, three main tabs: Forecast, SHAP, EDA.
  - Modern, vibrant plots (matplotlib, seaborn, plotly).
  - Download buttons for all graphs and images.
  - Enhanced prediction table with AQI level highlighting.
- **EDA Tools:**
  - Correlation heatmap, feature distributions, time-series trends.

---

## Directory Structure

```
.
├── app.py                        # Streamlit dashboard
├── backfill_data.py              # Automated data fetching & merging
├── predict.py                    # Model prediction & SHAP analysis
├── model.py                      # Model training (various regressors)
├── requirements.txt              # Python dependencies
├── data/
│   ├── predicted_aqi_72hr.csv    # Latest 72-hour forecast
│   ├── shap_summary_karachi.png  # SHAP plots (per city)
│   └── ...
├── model/
│   ├── RidgeRegression.joblib    # Trained models
│   └── ...
├── .github/
│   └── workflows/
│       ├── hourly_features.yml   # CI/CD for data update
│       └── update_forecast.yml   # CI/CD for forecast update
└── ...
```

---

## Quickstart

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd AQI_Prediction
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
   For Plotly image downloads, also install:
   ```sh
   pip install -U kaleido
   ```

3. **Run the dashboard:**
   ```sh
   streamlit run app.py
   ```

4. **Automated Data & Forecast Updates:**
   - Data and forecast are updated daily via GitHub Actions (see `.github/workflows/`).
   - To run manually:
     ```sh
     python backfill_data.py
     python predict.py
     ```

---

## Usage

- **City Selection:** Use the dropdown to select Karachi, Islamabad, or Lahore.
- **Tabs:**
  - **Forecast:** View 72-hour AQI forecast plot and enhanced prediction table. Download the plot as PNG.
  - **SHAP:** View and download SHAP summary plot for model explainability.
  - **EDA:** Explore correlation heatmap, feature distributions, and time-series trends. All plots are downloadable.

---

## Data Sources
- [Open-Meteo Air Quality API](https://open-meteo.com/)
- [Open-Meteo Weather API](https://open-meteo.com/)

---

## Key Python Dependencies
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn
- joblib
- requests
- openmeteo_requests
- requests_cache
- retry_requests
- shap
- kaleido (for Plotly image export)

---

## CI/CD
- Automated workflows for daily data and forecast updates using GitHub Actions.
- See `.github/workflows/hourly_features.yml` and `.github/workflows/update_forecast.yml`.

---

## Screenshots

> Add screenshots of the dashboard here for a visual overview.

---

## License

MIT License. See `LICENSE` file for details.

---

## Acknowledgements
- Open-Meteo for free air quality and weather data APIs.
- SHAP for model explainability tools.
- Streamlit, matplotlib, seaborn, and plotly for visualization.

---

## Contact

For questions or contributions, please open an issue or pull request on GitHub.
