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
import shap

# Setup logging
logging.basicConfig(level=logging.INFO, filename="data_fetch.log", 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration
city_info = [
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011, "tz": "Asia/Karachi", "city_code": 0},
    {"name": "Islamabad", "lat": 33.6844, "lon": 73.0479, "tz": "Asia/Karachi", "city_code": 1},
    {"name": "Lahore", "lat": 31.5497, "lon": 74.3436, "tz": "Asia/Karachi", "city_code": 2},
]
os.makedirs("data", exist_ok=True)

def fetch_forecast_data(city, openmeteo):
    """Fetch 72-hour weather and air quality forecast from Open-Meteo."""
    try:
        # Air Quality
        aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        aq_params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "hourly": ["pm2_5", "pm10", "ozone", "carbon_monoxide",
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
            "pm2_5": hourly.Variables(0).ValuesAsNumpy(),
            "pm10": hourly.Variables(1).ValuesAsNumpy(),
            "ozone": hourly.Variables(2).ValuesAsNumpy(),
            "carbon_monoxide": hourly.Variables(3).ValuesAsNumpy(),
            "nitrogen_dioxide": hourly.Variables(4).ValuesAsNumpy(),
            "sulphur_dioxide": hourly.Variables(5).ValuesAsNumpy()
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
        forecast_df["city"] = city["city_code"]
        forecast_df = forecast_df[(forecast_df["temperature_2m"] >= 0) & (forecast_df["temperature_2m"] <= 50)]
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
    
    for lag in [1, 6, 24]:
        df[f"pm2_5_lag_{lag}"] = df["pm2_5"].shift(lag) if "pm2_5" in df else np.nan
        df[f"pm10_lag_{lag}"] = df["pm10"].shift(lag) if "pm10" in df else np.nan
    
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

def generate_shap_plot(model, X_forecast_full, city_name, is_ridge=False):
    """Generate and save SHAP summary plot for the model predictions."""
    try:
        if is_ridge:
            # Use LinearExplainer for Ridge with background data
            background = X_forecast_full.iloc[:100]  # Use first 100 rows as background
            explainer = shap.LinearExplainer(model, background)
        else:
            # Use TreeExplainer for XGBoost
            explainer = shap.TreeExplainer(model)
        
        shap_values = explainer.shap_values(X_forecast_full)
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_forecast_full, show=False)
        plt.title(f"SHAP Feature Importance for {city_name}")
        shap_plot_file = f"data/shap_{city_name.lower()}.png"
        plt.savefig(shap_plot_file, dpi=150, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved SHAP plot for {city_name} to {shap_plot_file}")
        return shap_plot_file, shap_values
    except Exception as e:
        logging.error(f"Failed to generate SHAP plot for {city_name}: {str(e)}")
        return None, None

def get_top_features(shap_values, feature_names, n_top=3):
    """Extract top N features and their importance from SHAP values."""
    try:
        # Calculate mean absolute SHAP values for each feature
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Get indices of top features
        top_indices = np.argsort(mean_shap)[-n_top:][::-1]
        
        # Get feature names and importance values
        top_features = [feature_names[i] for i in top_indices]
        top_importance = [mean_shap[i] for i in top_indices]
        
        return top_features, top_importance
    except Exception as e:
        logging.error(f"Failed to extract top features: {str(e)}")
        return ["pm2_5", "pm10", "temperature_2m"], [0.1, 0.1, 0.1]  # Default fallback
    
def main():
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Load historical data for lagged/rolling features
    data_file = max(glob.glob("data/historical_combined_cities_*.csv"), key=os.path.getctime)
    df = pd.read_csv(data_file, parse_dates=["time"]).set_index("time")

    for city in city_info:
        # Fetch and process forecast data
        forecast_df = fetch_forecast_data(city, openmeteo)
        if forecast_df is None:
            print(f"No forecast data for {city['name']}. Check logs.")
            continue
        forecast_df = add_features(forecast_df, city)
        X_forecast = forecast_df

        # Load model and scaler based on city
        if city["name"] == "Karachi":
            model_file = f"model/Ridge_{city['name']}.joblib"
            scaler_file = f"model/scaler_{city['name']}.joblib"
            use_ridge = True
        else:
            model_file = f"model/XGBoost_{city['name']}.joblib"
            scaler_file = None
            use_ridge = False

        if not os.path.exists(model_file):
            print(f"No model found for {city['name']} at {model_file}. Skipping.")
            continue
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file) if scaler_file and os.path.exists(scaler_file) else None
        recent_data = df[df["city"] == city["city_code"]].tail(24)
        X_train_cols = pd.read_csv(data_file).drop(columns=["us_aqi", "time"] + [col for col in df.columns if "aqi_lag" in col]).columns

        # Initialize forecast features with correct dtypes
        dtypes = {col: float for col in X_train_cols}
        X_forecast_full = pd.DataFrame(index=X_forecast.index, columns=X_train_cols).astype(dtypes)

        # Populate forecast features
        for col in X_train_cols:
            if col in X_forecast.columns:
                X_forecast_full[col] = X_forecast[col]

        # Populate air quality and rolling features with historical data
        for feature in ["pm2_5", "pm10", "ozone", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide"]:
            if feature in X_forecast_full.columns:
                X_forecast_full[feature] = recent_data[feature].iloc[-1]
        for feature in ["pm2_5_rolling_mean_24h", "pm2_5_rolling_std_24h", "pm10_rolling_mean_24h", "pm10_rolling_std_24h"]:
            if feature in X_forecast_full.columns:
                X_forecast_full[feature] = recent_data[feature].iloc[-1]

        # Populate lagged features (only pm2_5 and pm10)
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

        # Fill NaNs
        X_forecast_full.ffill(inplace=True)
        X_forecast_full.fillna(0, inplace=True)

        # Predict AQI
        if use_ridge and scaler:
            X_forecast_scaled = scaler.transform(X_forecast_full)
            y_pred = model.predict(X_forecast_scaled)
        else:
            y_pred = model.predict(X_forecast_full)
        forecast_df["predicted_aqi"] = y_pred
        # print column name of forecast df
        # print(f"Forecast DataFrame columns: {forecast_df.columns.tolist()}")

        # Generate SHAP plot and get SHAP values
        shap_plot_file, shap_values = generate_shap_plot(model, X_forecast_full, city["name"], is_ridge=use_ridge)
        if shap_plot_file:
            print(f"Saved SHAP plot for {city['name'].lower()} to {shap_plot_file}")

        # Extract top features and their importance
        if shap_values is not None:
            top_features, top_importance = get_top_features(shap_values, X_forecast_full.columns)
            
            # Add top features and importance to forecast_df
            for i in range(3):
                forecast_df[f"top_feature_{i+1}"] = top_features[i] if i < len(top_features) else "pm2_5"
                forecast_df[f"top_feature_{i+1}_importance"] = top_importance[i] if i < len(top_importance) else 0.1
        else:
            # Fallback values if SHAP analysis fails
            for i in range(3):
                forecast_df[f"top_feature_{i+1}"] = ["pm2_5", "pm10", "temperature_2m"][i]
                forecast_df[f"top_feature_{i+1}_importance"] = 0.1

        # round all numeric columns to 2 decimal places
        forecast_df = forecast_df.round(2)

        # Save predictions with all features (including top features)
        forecast_file = f"data/predictions_{city['name']}.csv"
        forecast_df.to_csv(forecast_file)
        print(f"Saved predictions for {city['name']} to {forecast_file}")
        logging.info(f"Saved predictions for {city['name']} to {forecast_file}")


if __name__ == "__main__":
    main()