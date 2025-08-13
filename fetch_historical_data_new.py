import requests
import pandas as pd
import numpy as np
from datetime import datetime
import os
import openmeteo_requests
import requests_cache
from retry_requests import retry
import logging
from concurrent.futures import ThreadPoolExecutor
import json

# Setup logging
logging.basicConfig(level=logging.INFO, filename="data_fetch.log", 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration
city_info = [
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011, "tz": "Asia/Karachi", "city_code": 0},
    {"name": "Islamabad", "lat": 33.6844, "lon": 73.0479, "tz": "Asia/Karachi", "city_code": 1},
    {"name": "Lahore", "lat": 31.5497, "lon": 74.3436, "tz": "Asia/Karachi", "city_code": 2},
]
past_days = 82
os.makedirs("data", exist_ok=True)

def fetch_aq_data(city, past_days, retry_session):
    """Fetch historical air quality data from Open-Meteo."""
    try:
        aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        aq_params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "past_days": past_days,
            "hourly": ["us_aqi", "pm10", "pm2_5", "ozone", "carbon_monoxide",
                       "nitrogen_dioxide", "sulphur_dioxide"],
            "timezone": city["tz"],
            "forecast_days": 0
        }
        response = retry_session.get(aq_url, params=aq_params)
        response.raise_for_status()
        data = pd.DataFrame(response.json()["hourly"])
        if data.empty:
            logging.error(f"No hourly air quality data for {city['name']}")
            return None
        data["time"] = pd.to_datetime(data["time"]).dt.tz_localize(None)
        data.set_index("time", inplace=True)
        return data
    except Exception as e:
        logging.error(f"Failed to fetch air quality for {city['name']}: {str(e)}")
        return None

def fetch_weather_data(city, past_days, openmeteo):
    """Fetch historical weather data from Open-Meteo."""
    try:
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "past_days": past_days,
            "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                       "cloud_cover", "precipitation"],
            "timezone": city["tz"],
            "forecast_days": 0

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
        weather_df.set_index("time", inplace=True)
        weather_df = weather_df[(weather_df["temperature_2m"] >= 0) & (weather_df["temperature_2m"] <= 50)]
        return weather_df
    except Exception as e:
        logging.error(f"Failed to fetch weather for {city['name']}: {str(e)}")
        return None

def add_features(df, city):
    """Add feature engineering to the DataFrame."""
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
        df[f"us_aqi_lag_{lag}"] = df["us_aqi"].shift(lag) if "us_aqi" in df else np.nan
    
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

def process_city(city, past_days, openmeteo, retry_session):
    """Process historical data for a single city."""
    aq_data = fetch_aq_data(city, past_days, retry_session)
    if aq_data is None:
        return None
    weather_df = fetch_weather_data(city, past_days, openmeteo)
    if weather_df is None:
        return None
    
    combined_df = pd.merge(aq_data, weather_df, left_index=True, right_index=True, how="inner")
    if combined_df.empty:
        logging.warning(f"No overlapping data for {city['name']}")
        return None
    
    combined_df = add_features(combined_df, city)
    return combined_df

def main():
    print("Fetching historical data for all cities...")
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    with ThreadPoolExecutor(max_workers=3) as executor:
        all_cities = list(executor.map(
            lambda city: process_city(city, past_days, openmeteo, retry_session), city_info))
    all_cities = [df for df in all_cities if df is not None]
    if not all_cities:
        print("No historical data fetched. Check logs.")
        logging.error("No historical data fetched")
        return

    final_df = pd.concat(all_cities)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/historical_combined_cities_new.csv"
    final_df.to_csv(output_file)
    print(f"Saved historical data to {output_file}")

    # metadata = {
    #     "cities": [city["name"] for city in city_info],
    #     "past_days": past_days,
    #     "timestamp": timestamp,
    #     "features": list(final_df.columns)
    # }
    # meta_file = f"data/metadata_{timestamp}.json"
    # with open(meta_file, "w") as f:
    #     json.dump(metadata, f)
    # print(f"Saved metadata to {meta_file}")
    logging.info(f"Saved historical data to {output_file}")

if __name__ == "__main__":
    main()