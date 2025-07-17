import pandas as pd

# Load forecast CSV
df = pd.read_csv("historical_raw.csv", parse_dates=["time"])
df.set_index("time", inplace=True)

# -------------------------
# TIME-BASED FEATURES
# -------------------------
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek  # Monday=0
df["month"] = df.index.month

# -------------------------
# DERIVED FEATURES
# -------------------------

# 1. AQI Change Rate
# df["aqi_diff_1h"] = df["us_aqi"].diff()

# 2. 3-hour moving average of AQI
# df["us_aqi_ma3"] = df["us_aqi"].rolling(window=3).mean()

# 3. PM Ratio (PM2.5/PM10)
df["pm_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)  # avoid division by zero

# 4. Lag feature (previous hour AQI)
# df["us_aqi_lag1"] = df["us_aqi"].shift(1)

# Drop initial NaNs caused by diff, shift, rolling
df.dropna(inplace=True)

# -------------------------
# Save Features
# -------------------------
df.to_csv("historical_features.csv")
print("Saved engineered features to historical_features.csv")
