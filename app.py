import plotly.express as px
import matplotlib.dates as mdates
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load prediction data
df = pd.read_csv("data/predicted_aqi_72hr.csv")
df["time"] = pd.to_datetime(df["time"])

# City selection (default Karachi)
city_map = {0: "Karachi", 1: "Islamabad", 2: "Lahore"}
city_name_to_code = {v: k for k, v in city_map.items()}
city_select = st.selectbox("Select City", ["Karachi", "Islamabad", "Lahore"], index=0, key="city_select")
selected_city_code = city_name_to_code[city_select]
city_df = df[df["city"] == selected_city_code].copy()

# Set up Streamlit app
st.set_page_config(page_title="72-Hour AQI Forecast", layout="centered")
st.title(f"🌫️ 72-Hour Air Quality Forecast for {city_select}")

# Define AQI color bands
def get_aqi_color(aqi):
    if aqi <= 50: return "🟢 Good"
    elif aqi <= 100: return "🟡 Moderate"
    elif aqi <= 150: return "🟠 Unhealthy (Sensitive)"
    elif aqi <= 200: return "🔴 Unhealthy"
    elif aqi <= 300: return "🟣 Very Unhealthy"
    else: return "⚫ Hazardous"

def highlight_aqi(val):
        if val <= 50: return 'background-color: #c6f7d0; color: black'      # Green
        elif val <= 100: return 'background-color: #fffacc; color: black'   # Yellow
        elif val <= 150: return 'background-color: #ffd3a3; color: black'   # Orange
        elif val <= 200: return 'background-color: #ff9999; color: black'   # Red
        elif val <= 300: return 'background-color: #d6a5ff; color: white'   # Purple
        else: return 'background-color: #3a3a3a; color: white'              # Black


# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ("Home", "Forecast, SHAP & EDA Tabs")
)


if page == "Home":
    # --- Original Home Page ---
    # Show metrics
    latest_aqi = city_df.iloc[-1]["predicted_us_aqi"]
    st.metric("Latest Predicted AQI", f"{latest_aqi:.1f}", help=get_aqi_color(latest_aqi))

    if latest_aqi > 150:
        st.warning("🚨 Air quality may be hazardous in the coming hours!")

    # Plot forecast
    st.subheader(f"📈 AQI Forecast (Next 72 Hours) - {city_select}")
    fig, ax = plt.subplots()
    ax.plot(city_df["time"], city_df["predicted_us_aqi"], color="purple", marker="o")
    ax.set_ylabel("Predicted AQI")
    ax.set_xlabel("Time")
    ax.set_title(f"Predicted AQI Trend - {city_select}")
    ax.grid(True)
    st.pyplot(fig)

    # Show table
    # Round values for clarity
    display_df = city_df.copy()
    display_df = display_df.round(1)

    display_df["AQI Level"] = display_df["predicted_us_aqi"].apply(get_aqi_color)

    # Optional: reorder columns to show time, AQI, and level first
    cols_to_drop = ["city", "month", "hour", "dayofweek"]
    cols_to_show_first = ["time", "predicted_us_aqi", "AQI Level"]
    display_df = display_df.drop(columns=[col for col in cols_to_drop if col in display_df.columns])

    other_cols = [col for col in display_df.columns if col not in cols_to_show_first]
    display_df = display_df[cols_to_show_first + other_cols]

    # Apply color gradient to predicted AQI
    styled_df = display_df.style.format(precision=1).applymap(
        highlight_aqi, subset=["predicted_us_aqi"]
    )

    # Add a little padding between previous plot and title
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Title and table display
    st.subheader("📋 Prediction Table (Enhanced)")
    st.dataframe(styled_df, use_container_width=True, height=600, width=1200)




    # SHAP Explanation
    # st.subheader(f"🧠 Feature Importance (SHAP Summary Plot) - {city_select}")
    # import os
    # shap_img_path = f"data/shap_summary_{city_select.lower()}.png"
    # if os.path.exists(shap_img_path):
    #     st.image(shap_img_path, caption=f"Feature impact on AQI prediction for {city_select}", use_container_width=True)
    # else:
    #     st.info("SHAP plot not available yet. Please run the prediction script first.")


elif page == "Forecast, SHAP & EDA Tabs":
    # --- Tabs UI ---
    tab2, tab3 = st.tabs(["🧠 SHAP", "🔬 EDA"])

    # ---------------------- #
    # 🧠 SHAP Tab
    # ---------------------- #
    with tab2:
        st.subheader(f"SHAP Summary Plot - {city_select}")
        import os
        shap_img_path = f"data/shap_summary_{city_select.lower()}.png"
        if os.path.exists(shap_img_path):
            st.image(shap_img_path, caption=f"Feature impact on AQI prediction for {city_select}", use_container_width=True)
        else:
            st.info("SHAP plot not available yet. Please run the prediction script first.")

    # ---------------------- #
    # 🔬 EDA Tab
    # ---------------------- #
    with tab3:
        
        st.subheader("📊 Correlation Heatmap")

        # Exclude irrelevant or non-numeric columns
        exclude_cols = ["time", "timestamp", "city", "month", "AQI Level"]
        numeric_cols = city_df.select_dtypes(include="number").columns.difference(exclude_cols)

        # Compute correlation
        corr = city_df[numeric_cols].corr()

        # Create a mask to hide the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr,
            mask=mask,
            cmap="viridis",
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            square=True,
            cbar_kws={"shrink": 0.75},
            ax=ax
        )
        st.pyplot(fig)


        st.subheader("📉 Feature Distribution")
        feature_cols = city_df.columns.drop(["time", "AQI Level", "predicted_us_aqi", "city"], errors="ignore")
        dist_feat = st.selectbox("Select feature for distribution plot", feature_cols, key="dist_feat")
        if dist_feat:
            fig, ax = plt.subplots(figsize=(8, 4))
            n, bins, patches = ax.hist(
                city_df[dist_feat].dropna(),
                bins=30,
                edgecolor="white",
                alpha=0.85
            )
            # Set vibrant colors for each bar
            for patch, color in zip(patches, plt.cm.plasma(np.linspace(0.2, 0.8, len(patches)))):
                patch.set_facecolor(color)
            ax.set_title(f"Distribution of {dist_feat}", fontsize=15, color="#333333")
            ax.set_xlabel(dist_feat, fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)


    # st.subheader("📆 Time-Series Feature Trend")

    # # Drop unnecessary columns for selection
    # ts_cols = city_df.columns.drop(["AQI Level"], errors="ignore")
    # ts_feat = st.selectbox("Select feature for time-series trend", ts_cols, key="ts_feat")

    # if ts_feat:
    #     fig, ax = plt.subplots(figsize=(10, 4))  # Wider figure
        
    #     # Plot the selected feature
    #     ax.plot(city_df["time"], city_df[ts_feat], color="green")
        
    #     # Improve x-axis label formatting
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))  # e.g., Jul 26 \n 15:00
    #     ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))          # Tick every 6 hours
    #     fig.autofmt_xdate(rotation=45)                                      # Rotate labels
        
    #     # Labels and grid
    #     ax.set_xlabel("Time")
    #     ax.set_ylabel(ts_feat)
    #     ax.set_title(f"Trend of {ts_feat} over Time")
    #     ax.grid(True)
        
    #     st.pyplot(fig)


    st.subheader("📆 Time-Series Feature Trend")
    ts_cols = city_df.columns.drop(["AQI Level","time"], errors="ignore")
    ts_feat = st.selectbox("Select feature for time-series trend", ts_cols, key="ts_feat")

    if ts_feat:
        fig = px.line(city_df, x="time", y=ts_feat, title=f"Trend of {ts_feat} over Time", markers=True)
        fig.update_traces(line_color="green")
        fig.update_layout(xaxis_title="Time", yaxis_title=ts_feat, height=400)
        st.plotly_chart(fig, use_container_width=True)

