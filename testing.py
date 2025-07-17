# import requests
# import pandas as pd

# url = "https://api.open-meteo.com/v1/forecast"
# params = {
#     "latitude": 24.8607,
#     "longitude": 67.0011,
#     "past_days": 90,
#     "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover,precipitation",
#     "timezone": "Asia/Karachi"
# }

# resp = requests.get(url, params=params)
# data = resp.json()
# # print(data)
# df = pd.DataFrame(data["hourly"])
# df["time"] = pd.to_datetime(df["time"])
# df.dropna(inplace=True)
# df.set_index("time", inplace=True)
# print(df.head())


import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 24.8607,
    "longitude": 67.0011,
	"hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "cloud_cover", "precipitation"],
	"timezone": "auto",
	"past_days": 82,
	"forecast_days": 0
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()}{response.TimezoneAbbreviation()}")


# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(3).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(4).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["cloud_cover"] = hourly_cloud_cover
hourly_data["precipitation"] = hourly_precipitation

hourly_dataframe = pd.DataFrame(data = hourly_data)
print(hourly_dataframe.head())
# print mean of precipitation
print("Mean precipitation:", hourly_dataframe["precipitation"].mean())
# print length before and after dropping NaN values
print("Length before dropping NaN:", len(hourly_dataframe))
print("Length after dropping NaN:", len(hourly_dataframe.dropna()))