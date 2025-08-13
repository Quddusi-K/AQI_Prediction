import pandas as pd
import numpy as np
import joblib
import glob
import os
import openmeteo_requests
import requests_cache
from retry_requests import retry
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, filename="test_models.log", 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration
city_info = [
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011, "tz": "Asia/Karachi", "city_code": 0},
    {"name": "Islamabad", "lat": 33.6844, "lon": 73.0479, "tz": "Asia/Karachi", "city_code": 1},
    {"name": "Lahore", "lat": 31.5497, "lon": 74.3436, "tz": "Asia/Karachi", "city_code": 2},
]
os.makedirs("data", exist_ok=True)

def fetch_forecast_data(city, openmeteo):
    """Fetch 72-hour air quality and weather forecast from Open-Meteo."""
    try:
        # Air Quality
        aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        aq_params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "hourly": ["us_aqi", "pm2_5", "pm10", "ozone", "carbon_monoxide",
                       "nitrogen_dioxide", "sulphur_dioxide"],
            "forecast_days": 3,
            "timezone": city["tz"]
        }
        response = openmeteo.weather_api(aq_url, params=aq_params)
        hourly = response[0].Hourly()
        aq_data = {
            "time": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "us_aqi": hourly.Variables(0).ValuesAsNumpy(),
            "pm2_5": hourly.Variables(1).ValuesAsNumpy(),
            "pm10": hourly.Variables(2).ValuesAsNumpy(),
            "ozone": hourly.Variables(3).ValuesAsNumpy(),
            "carbon_monoxide": hourly.Variables(4).ValuesAsNumpy(),
            "nitrogen_dioxide": hourly.Variables(5).ValuesAsNumpy(),
            "sulphur_dioxide": hourly.Variables(6).ValuesAsNumpy()
        }
        aq_df = pd.DataFrame(aq_data)
        aq_df["time"] = aq_df["time"].dt.tz_localize(None)

        # Weather
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                       "cloud_cover", "precipitation"],
            "forecast_days": 3,
            "timezone": city["tz"]
        }
        responses = openmeteo.weather_api(weather_url, params=weather_params)
        hourly = responses[0].Hourly()
        weather_data = {
            "time": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(),
            "wind_speed_10m": hourly.Variables(2).ValuesAsNumpy(),
            "cloud_cover": hourly.Variables(3).ValuesAsNumpy(),
            "precipitation": hourly.Variables(4).ValuesAsNumpy()
        }
        weather_df = pd.DataFrame(weather_data)
        weather_df["time"] = weather_df["time"].dt.tz_localize(None)

        # Merge
        forecast_df = pd.merge(aq_df, weather_df, on="time")
        forecast_df.set_index("time", inplace=True)
        forecast_df = forecast_df[(forecast_df["temperature_2m"] >= 0) & (forecast_df["temperature_2m"] <= 50)]
        forecast_df["city"] = city["city_code"]
        return forecast_df
    except Exception as e:
        logging.error(f"Failed to fetch forecast for {city['name']}: {str(e)}")
        return None

def add_features(df, city):
    """Add feature engineering to the DataFrame, matching training features."""
    df["dayofweek"] = df.index.dayofweek
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["city"] = city["city_code"]
    
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    for lag in [1, 6, 24]:  # Include lags 1, 6, 24 for pm2_5, pm10, and us_aqi
        df[f"pm2_5_lag_{lag}"] = df["pm2_5"].shift(lag) if "pm2_5" in df else np.nan
        df[f"pm10_lag_{lag}"] = df["pm10"].shift(lag) if "pm10" in df else np.nan
        # df[f"us_aqi_lag_{lag}"] = df["us_aqi"].shift(lag) if "us_aqi" in df else np.nan


    if "pm2_5" in df:
        df["pm2_5_rolling_mean_24h"] = df["pm2_5"].rolling(window=24, min_periods=1).mean()
        df["pm2_5_rolling_std_24h"] = df["pm2_5"].rolling(window=24, min_periods=1).std()
    if "pm10" in df:
        df["pm10_rolling_mean_24h"] = df["pm10"].rolling(window=24, min_periods=1).mean()
        df["pm10_rolling_std_24h"] = df["pm10"].rolling(window=24, min_periods=1).std()
    
    df["temp_humidity_interaction"] = df["temperature_2m"] * df["relative_humidity_2m"]
    df["wind_pm25_interaction"] = df["wind_speed_10m"] * df.get("pm2_5", 1)
    
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    return df

def main():
    print("Testing model performance against Open-Meteo AQI forecasts...")
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Load historical data for lagged/rolling features
    data_file = max(glob.glob("data/historical_combined_cities_*.csv"), key=os.path.getctime)
    df = pd.read_csv(data_file, parse_dates=["time"]).set_index("time")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Define models
    model_names = ["Ridge", "RandomForest", "GradientBoosting", "XGBoost"]
    results = []

    for city in city_info:
        # Fetch forecast data
        forecast_df = fetch_forecast_data(city, openmeteo)
        if forecast_df is None:
            print(f"No forecast data for {city['name']}. Check logs.")
            continue
        y_true = forecast_df["us_aqi"].copy()
        forecast_df = add_features(forecast_df, city)
        X_forecast = forecast_df.drop(columns=["us_aqi"], errors="ignore")

        # Load historical data for feature imputation
        recent_data = df[df["city"] == city["city_code"]].tail(24)
        # drop all lag features
        X_train_cols = pd.read_csv(data_file).drop(columns=["us_aqi", "time", "us_aqi_lag_1", "us_aqi_lag_6"]).columns
        X_train_cols = X_train_cols.drop([col for col in X_train_cols if "lag" in col])

        # Initialize forecast features
        dtypes = {col: float for col in X_train_cols}
        X_forecast_full = pd.DataFrame(index=X_forecast.index, columns=X_train_cols).astype(dtypes)

        # Populate forecast features
        for col in X_train_cols:
            if col in X_forecast.columns:
                X_forecast_full[col] = X_forecast[col]

        # Populate air quality and rolling features
        for feature in ["pm2_5", "pm10", "ozone", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide"]:
            if feature in X_forecast_full.columns:
                X_forecast_full[feature] = recent_data[feature].iloc[-1]
        for feature in ["pm2_5_rolling_mean_24h", "pm2_5_rolling_std_24h", "pm10_rolling_mean_24h", "pm10_rolling_std_24h"]:
            if feature in X_forecast_full.columns:
                X_forecast_full[feature] = recent_data[feature].iloc[-1]

        # Populate lagged features (pm2_5, pm10 for lags 1, 6, 24)
        for lag in [1, 6, 24]:
            for feature in ["pm2_5", "pm10"]:
                col = f"{feature}_lag_{lag}"
                if col in X_forecast_full.columns:
                    lag_values = recent_data[feature].tail(lag).values
                    if len(lag_values) < lag:
                        lag_values = np.pad(lag_values, (lag - len(lag_values), 0), mode="edge")
                    forecast_values = np.full(len(X_forecast_full), lag_values[-1], dtype=float)
                    if lag <= len(X_forecast_full):
                        forecast_values[:lag] = lag_values[-lag:]
                    X_forecast_full[col] = forecast_values
        # Populate us_aqi_lag_24
        if "us_aqi_lag_24" in X_forecast_full.columns:
            lag_values = recent_data["us_aqi"].tail(24).values
            if len(lag_values) < 24:
                lag_values = np.pad(lag_values, (24 - len(lag_values), 0), mode="edge")
            forecast_values = np.full(len(X_forecast_full), lag_values[-1], dtype=float)
            if 24 <= len(X_forecast_full):
                forecast_values[:24] = lag_values[-24:]
            X_forecast_full["us_aqi_lag_24"] = forecast_values

        X_forecast_full.ffill(inplace=True)
        X_forecast_full.fillna(0, inplace=True)

        # Test each model
        predictions = {}
        for model_name in model_names:
            model = joblib.load(f"model/{model_name}_{city['name']}.joblib")
            scaler = joblib.load(f"model/scaler_{city['name']}.joblib") if os.path.exists(f"model/scaler_{city['name']}.joblib") else None
            
            # Apply scaling only for Ridge
            if model_name == "Ridge" and scaler:
                X_forecast_scaled = scaler.transform(X_forecast_full)
            else:
                X_forecast_scaled = X_forecast_full  # Keep DataFrame for other models
            
            y_pred = model.predict(X_forecast_scaled)
            predictions[model_name] = y_pred

            # Evaluate
            mae = mean_absolute_error(y_true, y_pred)
            rmse = root_mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            results.append({
                "City": city["name"],
                "Model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })

        # Plot comparison
        plt.figure(figsize=(8, 4))
        plt.plot(forecast_df.index, y_true, label="Open-Meteo AQI", color="black", linewidth=2)
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        for idx, model_name in enumerate(model_names):
            plt.plot(forecast_df.index, predictions[model_name], label=model_name, color=colors[idx])
        plt.legend()
        plt.title(f"AQI Forecast Comparison for {city['name']}")
        plt.xlabel("Time")
        plt.ylabel("US AQI")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_file = f"data/test_comparison_{city['name']}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved comparison plot for {city['name']} to {plot_file}")

    # Save results
    results_df = pd.DataFrame(results)
    results_file = f"data/test_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nSaved test results to {results_file}")
    print("\nModel Performance against Open-Meteo AQI:")
    print(results_df.sort_values(by=["City", "RMSE"]))
    logging.info(f"Saved test results to {results_file}")

if __name__ == "__main__":
    main()