
import os
import pandas as pd
from datetime import datetime, timedelta
import requests
import openmeteo_requests
import requests_cache
from retry_requests import retry

# --- Config ---
CITY_INFO = [
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011, "tz": "Asia/Karachi", "city_code": 0},
    {"name": "Islamabad", "lat": 33.6844, "lon": 73.0479, "tz": "Asia/Karachi", "city_code": 1},
    {"name": "Lahore", "lat": 31.5497, "lon": 74.3436, "tz": "Asia/Karachi", "city_code": 2},
]
HISTORICAL_PATH = "data/historical_combined_cities.csv"

# --- Date range: past 24 hours ---
end_date = datetime.now().date()
start_date = end_date - timedelta(days=1)

all_cities = []
for city in CITY_INFO:
    print(f"Fetching data for {city['name']} ({start_date} to {end_date})...")
    # Air Quality
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": [
            "us_aqi", "pm10", "pm2_5", "ozone", "carbon_monoxide",
            "nitrogen_dioxide", "sulphur_dioxide"
        ],
        "timezone": city["tz"]
    }
    aq_resp = requests.get(aq_url, params=aq_params)
    if aq_resp.status_code != 200:
        print(f"❌ Failed to fetch air quality for {city['name']}")
        continue
    aq_df = pd.DataFrame(aq_resp.json()["hourly"])
    aq_df["time"] = pd.to_datetime(aq_df["time"])
    aq_df["time"] = aq_df["time"].dt.tz_localize(None)
    aq_df.set_index("time", inplace=True)
    aq_df.sort_index(inplace=True)

    # Weather (use openmeteo_requests for robust fetching)
    print(f"Fetching weather data for {city['name']}...")
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
            "cloud_cover", "precipitation"
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
    merged = pd.merge(aq_df, weather_df, left_index=True, right_index=True, how="inner")
    merged.dropna(inplace=True)
    merged["dayofweek"] = merged.index.dayofweek
    merged["hour"] = merged.index.hour
    merged["month"] = merged.index.month
    merged["city"] = city["city_code"]
    all_cities.append(merged)


# Combine all cities' new data
if not all_cities:
    print("No new data fetched.")
    exit()
new_data = pd.concat(all_cities)

# Ensure 'time' is a column for overlap check
if "time" not in new_data.columns:
    new_data_reset = new_data.reset_index()
else:
    new_data_reset = new_data

# Load historical data if exists
if os.path.exists(HISTORICAL_PATH):
    hist = pd.read_csv(HISTORICAL_PATH)
    hist["time"] = pd.to_datetime(hist["time"])
    # Remove overlap: keep only rows in hist whose (time, city) not in new_data
    merged_keys = set(zip(new_data_reset["time"], new_data_reset["city"]))
    hist = hist[~hist.set_index(["time", "city"]).index.isin(merged_keys)]
    # Concatenate, new data last (priority)
    combined = pd.concat([hist, new_data_reset], ignore_index=True)
else:
    combined = new_data_reset.copy()

# Save
os.makedirs(os.path.dirname(HISTORICAL_PATH), exist_ok=True)
combined.to_csv(HISTORICAL_PATH, index=False)
print(f"✅ Updated {HISTORICAL_PATH} with latest 24h data.")
