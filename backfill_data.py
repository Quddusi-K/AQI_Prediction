import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# Karachi coordinates
latitude = 24.8607
longitude = 67.0011

# Time range
days_to_fetch = 90
end_date = datetime.now().date()
start_date = end_date - timedelta(days=days_to_fetch)

all_data = []

for day_offset in range(days_to_fetch):
    date = start_date + timedelta(days=day_offset)
    date_str = date.strftime('%Y-%m-%d')
    print(f"Fetching data for {date_str}")

    # --- Fetch AIR QUALITY data ---
    air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    air_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": [
            "us_aqi", "pm10", "pm2_5", "ozone", "carbon_monoxide",
            "nitrogen_dioxide", "sulphur_dioxide"
        ],
        "timezone": "Asia/Karachi"
    }
    air_resp = requests.get(air_url, params=air_params)
    
    # --- Fetch WEATHER data ---
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
            "cloud_cover", "precipitation"
        ],
        "timezone": "Asia/Karachi"
    }
    weather_resp = requests.get(weather_url, params=weather_params)

    if air_resp.status_code == 200 and weather_resp.status_code == 200:
        air_data = pd.DataFrame(air_resp.json()["hourly"])
        weather_data = pd.DataFrame(weather_resp.json()["hourly"])

        if not air_data.empty and not weather_data.empty:
            air_data["time"] = pd.to_datetime(air_data["time"])
            weather_data["time"] = pd.to_datetime(weather_data["time"])

            merged = pd.merge(air_data, weather_data, on="time", how="inner")
            all_data.append(merged)
    else:
        print(f"Failed on {date_str} → Air: {air_resp.status_code}, Weather: {weather_resp.status_code}")

# Save to CSV
os.makedirs("data", exist_ok=True)
df = pd.concat(all_data)
# Add time features
df["dayofweek"] = df["time"].dt.dayofweek
df["hour"] = df["time"].dt.hour
df["month"] = df["time"].dt.month
df.set_index("time", inplace=True)
df.to_csv("data/historical_raw.csv")
print("Saved past 90-day weather + AQI data to data/historical_raw.csv with dayofweek, hour, month columns.")
