import pandas as pd
import requests
import joblib
from datetime import datetime
import shap
import matplotlib.pyplot as plt
from os import makedirs


# Load best model
model = joblib.load("model/RidgeRegression.joblib")

# City info: Karachi=0, Islamabad=1, Lahore=2
city_info = [
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011, "tz": "Asia/Karachi", "city_code": 0},
    {"name": "Islamabad", "lat": 33.6844, "lon": 73.0479, "tz": "Asia/Karachi", "city_code": 1},
    {"name": "Lahore", "lat": 31.5497, "lon": 74.3436, "tz": "Asia/Karachi", "city_code": 2},
]


# Collect predictions for all cities
all_city_results = []
for city in city_info:
    print(f"Fetching forecast for {city['name']}...")
    # Air Quality
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "hourly": [
            "us_aqi", "pm10", "pm2_5", "ozone",
            "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide"
        ],
        "forecast_days": 3,
        "timezone": city["tz"]
    }
    aq_response = requests.get(aq_url, params=aq_params)
    if aq_response.status_code != 200:
        print(f"❌ Failed to fetch air quality forecast for {city['name']}")
        continue
    aq_df = pd.DataFrame(aq_response.json()["hourly"])

    # Weather
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "hourly": [
            "temperature_2m", "relative_humidity_2m",
            "wind_speed_10m", "cloud_cover", "precipitation"
        ],
        "forecast_days": 3,
        "timezone": city["tz"]
    }
    weather_response = requests.get(weather_url, params=weather_params)
    if weather_response.status_code != 200:
        print(f"❌ Failed to fetch weather forecast for {city['name']}")
        continue
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
    # Add city column
    df["city"] = city["city_code"]
    # Predict
    predictions = model.predict(df)
    df["predicted_us_aqi"] = predictions
    all_city_results.append(df)

# Combine all cities
all_results = pd.concat(all_city_results)


# SHAP analysis for each city
for city, city_df in zip(city_info, all_city_results):
    explainer = shap.Explainer(model, city_df.drop(columns=["predicted_us_aqi"]))
    shap_values = explainer(city_df.drop(columns=["predicted_us_aqi"]))
    plt.figure()
    shap.summary_plot(shap_values, city_df.drop(columns=["predicted_us_aqi"]), show=False)
    plt.tight_layout()
    plt.savefig(f"data/shap_summary_{city['name'].lower()}.png", dpi=300)
    plt.close()
    
makedirs("data", exist_ok=True)

# Save results with all features and predictions
all_results.reset_index(inplace=True)
all_results.to_csv("data/predicted_aqi_72hr.csv", index=False)
print("✅ Saved 72-hour AQI predictions for all cities to data/predicted_aqi_72hr.csv")
print("SHAP summary plots saved for each city")
