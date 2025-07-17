import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load prediction data
df = pd.read_csv("data/predicted_aqi_72hr.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Set up Streamlit app
st.set_page_config(page_title="72-Hour AQI Forecast", layout="centered")
st.title("🌫️ 72-Hour Air Quality Forecast for Karachi")

# Define AQI color bands
def get_aqi_color(aqi):
    if aqi <= 50: return "🟢 Good"
    elif aqi <= 100: return "🟡 Moderate"
    elif aqi <= 150: return "🟠 Unhealthy (Sensitive)"
    elif aqi <= 200: return "🔴 Unhealthy"
    elif aqi <= 300: return "🟣 Very Unhealthy"
    else: return "⚫ Hazardous"

# Show metrics
latest_aqi = df.iloc[-1]["predicted_us_aqi"]
st.metric("Latest Predicted AQI", f"{latest_aqi:.1f}", help=get_aqi_color(latest_aqi))

# Optional warning
if latest_aqi > 150:
    st.warning("🚨 Air quality may be hazardous in the coming hours!")

# Plot forecast
st.subheader("📈 AQI Forecast (Next 72 Hours)")
fig, ax = plt.subplots()
ax.plot(df["timestamp"], df["predicted_us_aqi"], color="purple", marker="o")
ax.set_ylabel("Predicted AQI")
ax.set_xlabel("Time")
ax.set_title("Predicted AQI Trend")
ax.grid(True)
st.pyplot(fig)

# Show table
st.subheader("📋 Prediction Table")
df["AQI Level"] = df["predicted_us_aqi"].apply(lambda x: get_aqi_color(x))
st.dataframe(df.style.highlight_max("predicted_us_aqi", color="red"))
