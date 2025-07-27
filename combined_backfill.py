import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import openmeteo_requests
import requests_cache
from retry_requests import retry


# -----------------------------
# Configuration
# -----------------------------
city_info = [
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011, "tz": "Asia/Karachi", "city_code": 0},
    {"name": "Islamabad", "lat": 33.6844, "lon": 73.0479, "tz": "Asia/Karachi", "city_code": 1},
    {"name": "Lahore", "lat": 31.5497, "lon": 74.3436, "tz": "Asia/Karachi", "city_code": 2},
]
past_days = 82

# Ensure output directory exists
os.makedirs("data", exist_ok=True)


# -----------------------------
# Fetch and combine for all cities
# -----------------------------
all_cities = []
for city in city_info:
    print(f"🔄 Fetching data for {city['name']}...")
    # Air Quality
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "past_days": past_days,
        "hourly": [
            "us_aqi", "pm10", "pm2_5", "ozone", "carbon_monoxide",
            "nitrogen_dioxide", "sulphur_dioxide"
        ],
        "timezone": city["tz"]
    }
    aq_response = requests.get(aq_url, params=aq_params)
    if aq_response.status_code != 200:
        print(f"❌ Failed to fetch air quality data for {city['name']}: {aq_response.status_code}")
        continue
    aq_data = pd.DataFrame(aq_response.json()["hourly"])
    aq_data["time"] = pd.to_datetime(aq_data["time"])
    aq_data["time"] = aq_data["time"].dt.tz_localize(None)
    aq_data.set_index("time", inplace=True)
    aq_data.sort_index(inplace=True)

    # Weather
    print(f"🔄 Fetching weather data for {city['name']}...")
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "past_days": past_days,
        "hourly": [
            "temperature_2m", "relative_humidity_2m",
            "wind_speed_10m", "cloud_cover", "precipitation"
        ],
        "timezone": city["tz"]
    }
    responses = openmeteo.weather_api(weather_url, params=weather_params)
    weather = responses[0]
    hourly = weather.Hourly()
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
    weather_df.sort_index(inplace=True)

    # Merge
    combined_df = pd.merge(aq_data, weather_df, left_index=True, right_index=True, how="inner")
    combined_df.dropna(inplace=True)
    # Add time features
    combined_df["dayofweek"] = combined_df.index.dayofweek
    combined_df["hour"] = combined_df.index.hour
    combined_df["month"] = combined_df.index.month
    # Add city column
    combined_df["city"] = city["city_code"]
    all_cities.append(combined_df)

# Concatenate all cities and save
final_df = pd.concat(all_cities) 

final_df.to_csv("data/historical_combined_cities.csv")
print("✅ Saved AQI + weather data for all cities to data/historical_combined_cities.csv")
