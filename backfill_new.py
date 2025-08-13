import pandas as pd
import numpy as np
import os
import openmeteo_requests
import requests_cache
from retry_requests import retry
import logging
import glob
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO, filename="backfill_data.log", 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration
city_info = [
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011, "tz": "Asia/Karachi", "city_code": 0},
    {"name": "Islamabad", "lat": 33.6844, "lon": 73.0479, "tz": "Asia/Karachi", "city_code": 1},
    {"name": "Lahore", "lat": 31.5497, "lon": 74.3436, "tz": "Asia/Karachi", "city_code": 2},
]
os.makedirs("data", exist_ok=True)

def fetch_aq_data(city, start_date, end_date, retry_session):
    """Fetch 24-hour air quality data from Open-Meteo."""
    try:
        aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        aq_params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
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

def fetch_weather_data(city, start_date, end_date, openmeteo):
    """Fetch 24-hour weather data from Open-Meteo."""
    try:
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
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

def process_city(city, start_date, end_date, openmeteo, retry_session):
    """Process 24-hour data for a single city."""
    aq_data = fetch_aq_data(city, start_date, end_date, retry_session)
    if aq_data is None:
        return None
    weather_df = fetch_weather_data(city, start_date, end_date, openmeteo)
    if weather_df is None:
        return None
    
    combined_df = pd.merge(aq_data, weather_df, left_index=True, right_index=True, how="inner")
    if combined_df.empty:
        logging.warning(f"No overlapping data for {city['name']}")
        return None
    
    combined_df = add_features(combined_df, city)
    return combined_df

def main():
    print("Backfilling 24-hour data for all cities...")
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Date range: past 24 hours
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=1)

    # Fetch new data in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        all_cities = list(executor.map(
            lambda city: process_city(city, start_date, end_date, openmeteo, retry_session), city_info))
    all_cities = [df for df in all_cities if df is not None]
    if not all_cities:
        print("No new data fetched. Check logs.")
        logging.error("No new data fetched")
        return

    new_data = pd.concat(all_cities)
    new_data_reset = new_data.reset_index()

    # Save 24-hour data to separate CSV
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # backfill_file = f"data/backfill_24h_{timestamp}.csv"
    # new_data_reset.to_csv(backfill_file, index=False)
    # print(f"Saved 24-hour data to {backfill_file}")
    # logging.info(f"Saved 24-hour data to {backfill_file}")

    # Load latest historical data
    hist_files = glob.glob("data/historical_combined_cities_new.csv")
    if hist_files:
        hist_file = max(hist_files, key=os.path.getctime)
        hist = pd.read_csv(hist_file, parse_dates=["time"])
    else:
        hist = pd.DataFrame(columns=new_data_reset.columns)

    # Remove duplicates based on time and city
    merged_keys = set(zip(new_data_reset["time"], new_data_reset["city"]))
    hist = hist[~hist[["time", "city"]].apply(tuple, axis=1).isin(merged_keys)]

    # Concatenate and save to historical CSV
    combined = pd.concat([hist, new_data_reset], ignore_index=True)
    output_file = f"data/historical_combined_cities_new.csv"
    combined.to_csv(output_file, index=False)
    print(f"Saved updated historical data to {output_file}")
    logging.info(f"Saved updated historical data to {output_file}")

if __name__ == "__main__":
    main()