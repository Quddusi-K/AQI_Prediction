import plotly.express as px
import matplotlib.dates as mdates
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO

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


# elif page == "SHAP & EDA Tabs":

# --- Three Tabs UI ---
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🧠 SHAP", "🔬 EDA"])

# --- Forecast Tab ---
with tab1:
    # Show metrics
    latest_aqi = city_df.iloc[-1]["predicted_us_aqi"]
    st.metric("Latest Predicted AQI", f"{latest_aqi:.1f}", help=get_aqi_color(latest_aqi))

    if latest_aqi > 150:
        st.warning("🚨 Air quality may be hazardous in the coming hours!")

    # Plot forecast
    st.subheader(f"📈 AQI Forecast (Next 72 Hours) - {city_select}")
    fig, ax = plt.subplots(figsize=(10, 5))
    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors
    import matplotlib as mpl
    x = city_df["time"].values
    y = city_df["predicted_us_aqi"].values
    x_num = mpl.dates.date2num(city_df["time"])
    points = np.array([x_num, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = mcolors.Normalize(vmin=min(y), vmax=max(y))
    lc = LineCollection(segments, cmap="plasma", norm=norm)
    lc.set_array(y)
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    scatter = ax.scatter(city_df["time"], y, c=y, cmap="plasma", s=80, edgecolor="white", zorder=3)
    cbar = fig.colorbar(line, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Predicted AQI", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    fig.autofmt_xdate(rotation=45)
    ax.set_ylabel("Predicted AQI", fontsize=13)
    ax.set_xlabel("Time", fontsize=13)
    ax.set_title(f"Predicted AQI Trend - {city_select}", fontsize=16, color="#333333", pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_facecolor("#f7f7fa")
    fig.patch.set_facecolor("#f7f7fa")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Download button for forecast plot
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    st.pyplot(fig)
    st.download_button(
        label="Download Forecast Plot as PNG",
        data=buf.getvalue(),
        file_name=f"aqi_forecast_{city_select.lower()}.png",
        mime="image/png"
    )

    # Show table
    display_df = city_df.copy()
    display_df = display_df.round(1)
    display_df["AQI Level"] = display_df["predicted_us_aqi"].apply(get_aqi_color)
    cols_to_drop = ["city", "month", "hour", "dayofweek"]
    cols_to_show_first = ["time", "predicted_us_aqi", "AQI Level"]
    display_df = display_df.drop(columns=[col for col in cols_to_drop if col in display_df.columns])
    other_cols = [col for col in display_df.columns if col not in cols_to_show_first]
    display_df = display_df[cols_to_show_first + other_cols]
    styled_df = display_df.style.format(precision=1).map(
        highlight_aqi, subset=["predicted_us_aqi"]
    )
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Prediction Table (Enhanced)")
    st.dataframe(styled_df, use_container_width=True, height=600, width=1200)

# --- SHAP Tab ---
with tab2:
    st.subheader(f"SHAP Summary Plot - {city_select}")
    import os
    shap_img_path = f"data/shap_summary_{city_select.lower()}.png"
    if os.path.exists(shap_img_path):
        st.image(shap_img_path, caption=f"Feature impact on AQI prediction for {city_select}", use_container_width=True)
        with open(shap_img_path, "rb") as img_file:
            st.download_button(
                label="Download SHAP Summary Plot",
                data=img_file,
                file_name=f"shap_summary_{city_select.lower()}.png",
                mime="image/png"
            )
    else:
        st.info("SHAP plot not available yet. Please run the prediction script first.")

# --- EDA Tab ---
with tab3:
    st.subheader("📊 Correlation Heatmap")
    exclude_cols = ["time", "timestamp", "city", "month", "AQI Level"]
    numeric_cols = city_df.select_dtypes(include="number").columns.difference(exclude_cols)
    corr = city_df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
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

    # Download button for correlation heatmap
    buf_corr = BytesIO()
    fig.savefig(buf_corr, format="png", bbox_inches="tight")
    st.pyplot(fig)
    st.download_button(
        label="Download Correlation Heatmap as PNG",
        data=buf_corr.getvalue(),
        file_name=f"correlation_heatmap_{city_select.lower()}.png",
        mime="image/png"
    )

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
        for patch, color in zip(patches, plt.cm.plasma(np.linspace(0.2, 0.8, len(patches)))):
            patch.set_facecolor(color)
        ax.set_title(f"Distribution of {dist_feat}", fontsize=15, color="#333333")
        ax.set_xlabel(dist_feat, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        buf_dist = BytesIO()
        fig.savefig(buf_dist, format="png", bbox_inches="tight")
        st.pyplot(fig)
        st.download_button(
            label=f"Download Distribution Plot as PNG",
            data=buf_dist.getvalue(),
            file_name=f"distribution_{dist_feat}_{city_select.lower()}.png",
            mime="image/png"
        )

    st.subheader("📆 Time-Series Feature Trend")
    ts_cols = city_df.columns.drop(["AQI Level","time"], errors="ignore")
    ts_feat = st.selectbox("Select feature for time-series trend", ts_cols, key="ts_feat")
    if ts_feat:
        fig = px.line(city_df, x="time", y=ts_feat, title=f"Trend of {ts_feat} over Time", markers=True)
        fig.update_traces(line_color="green")
        fig.update_layout(xaxis_title="Time", yaxis_title=ts_feat, height=400)
        st.plotly_chart(fig, use_container_width=True)
        # Download button for plotly figure
        try:
            img_bytes = fig.to_image(format="png")
            st.download_button(
                label=f"Download Time-Series Trend as PNG",
                data=img_bytes,
                file_name=f"timeseries_{ts_feat}_{city_select.lower()}.png",
                mime="image/png"
            )
        except Exception:
            st.info("Plotly image export requires kaleido. Install with: pip install -U kaleido")

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


