import pandas as pd
import requests
import joblib
from datetime import datetime

# Load best model
model = joblib.load("model/RidgeRegression.joblib")

# ---------------------------
# API: Air Quality Forecast
# ---------------------------
aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
aq_params = {
    "latitude": 24.8607,
    "longitude": 67.0011,
    "hourly": [
        "us_aqi", "pm10", "pm2_5", "ozone",
        "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide"
    ],
    "forecast_days": 3,
    "timezone": "Asia/Karachi"
}

# ---------------------------
# API: Weather Forecast
# ---------------------------
weather_url = "https://api.open-meteo.com/v1/forecast"
weather_params = {
    "latitude": 24.8607,
    "longitude": 67.0011,
    "hourly": [
        "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "cloud_cover", "precipitation"
    ],
    "forecast_days": 3,
    "timezone": "Asia/Karachi"
}

# Fetch data
aq_response = requests.get(aq_url, params=aq_params)
weather_response = requests.get(weather_url, params=weather_params)

# Error checks
if aq_response.status_code != 200:
    raise Exception("❌ Failed to fetch air quality forecast")

if weather_response.status_code != 200:
    raise Exception("❌ Failed to fetch weather forecast")

# Parse hourly JSON
aq_df = pd.DataFrame(aq_response.json()["hourly"])
weather_df = pd.DataFrame(weather_response.json()["hourly"])

# Merge by time
df = pd.merge(aq_df, weather_df, on="time")
df["time"] = pd.to_datetime(df["time"])
df.set_index("time", inplace=True)
df.sort_index(inplace=True)

# Drop us_aqi (target) if present
if "us_aqi" in df.columns:
    df.drop(columns=["us_aqi"], inplace=True)

# Add time-based features
df["dayofweek"] = df.index.dayofweek
df["hour"] = df.index.hour
df["month"] = df.index.month

# Predict
predictions = model.predict(df)

# Save results
output = pd.DataFrame({
    "timestamp": df.index,
    "predicted_us_aqi": predictions
})
output.to_csv("data/predicted_aqi_72hr.csv", index=False)
print("✅ Saved 72-hour AQI predictions to data/predicted_aqi_72hr.csv")
