import matplotlib.dates as mdates
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import glob
import os
import plotly.express as px
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib as mpl

# Columns in prediction csv: time,pm2_5,pm10,ozone,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover,precipitation,city,dayofweek,hour,month,hour_sin,hour_cos,month_sin,month_cos,pm2_5_lag_1,pm10_lag_1,pm2_5_lag_6,pm10_lag_6,pm2_5_lag_24,pm10_lag_24,pm2_5_rolling_mean_24h,pm2_5_rolling_std_24h,pm10_rolling_mean_24h,pm10_rolling_std_24h,temp_humidity_interaction,wind_pm25_interaction,predicted_aqi,top_feature_1,top_feature_1_importance,top_feature_2,top_feature_2_importance,top_feature_3,top_feature_3_importance

# Load prediction data
city_map = {0: "Karachi", 1: "Islamabad", 2: "Lahore"}
city_name_to_code = {v: k for k, v in city_map.items()}
city_select = st.selectbox("Select City", ["Karachi", "Islamabad", "Lahore"], index=0, key="city_select")
selected_city_code = city_name_to_code[city_select]

# Find the latest prediction CSV for the selected city
pred_files = glob.glob(f"data/predictions_{city_select}.csv")
if not pred_files:
    st.error(f"No prediction data found for {city_select}. Please run forecast_aqi.py first.")
    st.stop()
pred_file = max(pred_files, key=os.path.getctime)
df = pd.read_csv(pred_file, parse_dates=["time"])
city_df = df[df["city"] == selected_city_code].copy()

# Set up Streamlit app
st.set_page_config(page_title="72-Hour AQI Forecast", layout="centered")
st.title(f"🌫️ 72-Hour Air Quality Forecast for {city_select}")

# Define AQI color bands
def get_aqi_color(aqi):
    if aqi <= 50: return "🟢 Good"
    elif aqi <= 100: return "🟡 Moderate"
    elif aqi <= 150: return "🟠 Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "🔴 Unhealthy"
    elif aqi <= 300: return "🟣 Very Unhealthy"
    else: return "⚫ Hazardous"

def highlight_aqi(val):
    if val <= 50: return 'background-color: #c6f7d0; color: black'  # Green
    elif val <= 100: return 'background-color: #fffacc; color: black'  # Yellow
    elif val <= 150: return 'background-color: #ffd3a3; color: black'  # Orange
    elif val <= 200: return 'background-color: #ff9999; color: black'  # Red
    elif val <= 300: return 'background-color: #d6a5ff; color: white'  # Purple
    else: return 'background-color: #3a3a3a; color: white'  # Black

# Four Tabs UI
tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "🧠 SHAP", "🔬 EDA", "🩺 Health Information"])

# Forecast Tab
with tab1:
    # Show metrics
    latest_aqi = city_df["predicted_aqi"].iloc[-1]
    st.metric("Latest Predicted AQI", f"{latest_aqi:.1f}")
    
    if latest_aqi > 150:
        st.warning("🚨 Air quality may be hazardous in the coming hours!")

    # Plot forecast
    st.subheader(f"📈 AQI Forecast (Next 72 Hours) - {city_select}")
    fig, ax = plt.subplots(figsize=(12, 6))
    x = city_df["time"].values
    y = city_df["predicted_aqi"].values
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
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%I:%M %p"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    fig.autofmt_xdate(rotation=45)
    ax.set_ylabel("Predicted AQI", fontsize=12)
    ax.set_xlabel("Time", fontsize=12)
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
    st.subheader("📋 Prediction Table")
    display_df = city_df[["time", "predicted_aqi", "temperature_2m", "relative_humidity_2m", "wind_speed_10m"]].copy()
    
    # Format time to 12-hour format for display
    display_df["time"] = display_df["time"].dt.strftime("%b %d, %I:%M %p")
    
    display_df = display_df.round(1)
    styled_df = display_df.style.format(precision=1).map(
        highlight_aqi, subset=["predicted_aqi"]
    )
    st.dataframe(styled_df, use_container_width=True, height=600)

# SHAP Tab
with tab2:
    st.subheader(f"SHAP Analysis - {city_select}")
    shap_img_path = max(glob.glob(f"data/shap_{city_select.lower()}.png"), key=os.path.getctime)
    if os.path.exists(shap_img_path):
        st.image(shap_img_path, caption=f"Feature impact on AQI prediction for {city_select}", use_container_width=True)
        with open(shap_img_path, "rb") as img_file:
            st.download_button(
                label="Download SHAP Summary Plot",
                data=img_file,
                file_name=f"shap_{city_select.lower()}.png",
                mime="image/png"
            )
        
        # Interactive SHAP feature importance
        st.subheader("Top Influential Features")
        top_features = city_df[["top_feature_1", "top_feature_1_importance",
                                "top_feature_2", "top_feature_2_importance",
                                "top_feature_3", "top_feature_3_importance"]].iloc[0]
        feature_df = pd.DataFrame({
            "Feature": [top_features["top_feature_1"], top_features["top_feature_2"], top_features["top_feature_3"]],
            "Importance": [top_features["top_feature_1_importance"], top_features["top_feature_2_importance"], top_features["top_feature_3_importance"]]
        })
        fig = px.bar(feature_df, x="Feature", y="Importance", title="Top 3 Feature Importance (SHAP)")
        fig.update_traces(marker_color="purple")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("SHAP plot not available yet. Please run forecast_aqi.py first.")

# EDA Tab
with tab3:
    st.subheader("📊 Correlation Heatmap")
    
    # Add options for heatmap customization
    col1, col2, col3 = st.columns(3)
    with col1:
        show_annotations = st.checkbox("Show correlation values", value=False, key="show_annot")
    with col2:
        correlation_threshold = st.slider("Min correlation to show", 0.0, 1.0, 0.3, 0.1, key="corr_thresh")
    with col3:
        max_features = st.selectbox("Max features to show", [10, 15, 20, 25, 30], index=2, key="max_feat")
    
    # Filter columns more aggressively
    exclude_cols = ["time", "top_feature_1", "top_feature_2", "top_feature_3",
                    "top_feature_1_importance", "top_feature_2_importance", "top_feature_3_importance",
                    "hour_cos", "hour_sin", "month_cos", "month_sin", "dayofweek", "hour", "month"]
    
    numeric_cols = city_df.select_dtypes(include="number").columns.difference(exclude_cols)
    
    # Limit number of features if too many
    if len(numeric_cols) > max_features:
        # Select features with highest variance or most important ones
        feature_variance = city_df[numeric_cols].var().sort_values(ascending=False)
        numeric_cols = feature_variance.head(max_features).index.tolist()
        st.info(f"Showing top {max_features} features with highest variance. Use slider above to adjust.")
    
    corr = city_df[numeric_cols].corr()
    
    # Apply correlation threshold filter
    if correlation_threshold > 0:
        # Create mask for low correlations
        low_corr_mask = np.abs(corr) < correlation_threshold
        # Combine with upper triangle mask
        mask = np.triu(np.ones_like(corr, dtype=bool)) | low_corr_mask
    else:
        mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create better sized figure
    fig_size = max(8, len(numeric_cols) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    # Create heatmap with better formatting
    heatmap = sns.heatmap(
        corr,
        mask=mask,
        cmap="RdBu_r",  # Better color scheme for correlations
        vmin=-1,
        vmax=1,
        center=0,
        annot=show_annotations,
        fmt=".2f" if show_annotations else "",
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        ax=ax,
        annot_kws={"size": 8} if show_annotations else {}
    )
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add title
    plt.title(f"Feature Correlation Matrix - {city_select}", pad=20, fontsize=14, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    buf_corr = BytesIO()
    fig.savefig(buf_corr, format="png", bbox_inches="tight", dpi=150)
    st.pyplot(fig)
    
    # Show correlation statistics
    st.subheader("📈 Correlation Insights")
    col1, col2, col3 = st.columns(3)
    
    # Find strongest correlations
    corr_pairs = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            corr_val = corr.iloc[i, j]
            if abs(corr_val) >= correlation_threshold:
                corr_pairs.append((corr.columns[i], corr.columns[j], corr_val))
    
    # Sort by absolute correlation value
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    with col1:
        st.metric("Total Features", len(numeric_cols))
    with col2:
        st.metric("Strong Correlations (≥0.5)", len([p for p in corr_pairs if abs(p[2]) >= 0.5]))
    with col3:
        st.metric("Moderate Correlations (≥0.3)", len(corr_pairs))
    
    # Show top correlations
    if corr_pairs:
        st.subheader("🔗 Top Correlations")
        top_corr_df = pd.DataFrame(corr_pairs[:10], columns=["Feature 1", "Feature 2", "Correlation"])
        top_corr_df["Correlation"] = top_corr_df["Correlation"].round(3)
        st.dataframe(top_corr_df, use_container_width=True)
    
    st.download_button(
        label="Download Correlation Heatmap as PNG",
        data=buf_corr.getvalue(),
        file_name=f"correlation_heatmap_{city_select.lower()}.png",
        mime="image/png"
    )

    st.subheader("📉 Feature Distribution")
    feature_cols = city_df.columns.drop(["time", "top_feature_1", "top_feature_2", "top_feature_3",
                                         "top_feature_1_importance", "top_feature_2_importance",
                                         "top_feature_3_importance"], errors="ignore")
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
    ts_cols = city_df.columns.drop(["top_feature_1", "top_feature_2", "top_feature_3",
                                    "top_feature_1_importance", "top_feature_2_importance",
                                    "top_feature_3_importance"], errors="ignore")
    ts_feat = st.selectbox("Select feature for time-series trend", ts_cols, key="ts_feat")
    if ts_feat:
        fig = px.line(city_df, x="time", y=ts_feat, title=f"Trend of {ts_feat} over Time", markers=True)
        fig.update_traces(line_color="green")
        fig.update_layout(xaxis_title="Time", yaxis_title=ts_feat, height=400)
        st.plotly_chart(fig, use_container_width=True)
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

# Health Recommendations Tab
with tab4:
    st.subheader("🩺 AQI Health Information")
    
    # Define AQI categories and health information
    def get_aqi_info(aqi):
        if aqi <= 50:
            return "🟢 Good", "Air quality is considered satisfactory, and air pollution poses little or no risk."
        elif aqi <= 100:
            return "🟡 Moderate", "Air quality is acceptable; however, some pollutants may be a concern for a small number of people."
        elif aqi <= 150:
            return "🟠 Unhealthy for Sensitive Groups", "Members of sensitive groups may experience health effects. The general public is not likely to be affected."
        elif aqi <= 200:
            return "🔴 Unhealthy", "Everyone may begin to experience health effects; members of sensitive groups may experience more serious effects."
        elif aqi <= 300:
            return "🟣 Very Unhealthy", "Health warnings of emergency conditions. The entire population is more likely to be affected."
        else:
            return "⚫ Hazardous", "Health alert: everyone may experience more serious health effects."
    
    latest_aqi = city_df["predicted_aqi"].iloc[-1]
    latest_category, latest_info = get_aqi_info(latest_aqi)
    
    st.metric("Current AQI Category", latest_category)
    st.info(latest_info)
    
    # Show AQI trend over time
    st.subheader("AQI Trend Over Time")
    aqi_df = city_df[["time", "predicted_aqi"]].copy()
    aqi_df["time"] = aqi_df["time"].dt.strftime("%b %d, %I:%M %p")
    
    # Add category information
    aqi_df["category"] = aqi_df["predicted_aqi"].apply(lambda x: get_aqi_info(x)[0])
    
    fig = px.scatter(aqi_df, x="time", y="predicted_aqi", color="category",
                     title="AQI Trend and Categories Over Time", 
                     labels={"predicted_aqi": "Predicted AQI"},
                     height=400)
    fig.update_traces(marker=dict(size=10))
    fig.update_layout(xaxis_title="Time", yaxis_title="Predicted AQI")
    st.plotly_chart(fig, use_container_width=True)
    
    # Show AQI statistics
    st.subheader("AQI Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Minimum AQI", f"{aqi_df['predicted_aqi'].min():.1f}")
    with col2:
        st.metric("Maximum AQI", f"{aqi_df['predicted_aqi'].max():.1f}")
    with col3:
        st.metric("Average AQI", f"{aqi_df['predicted_aqi'].mean():.1f}")
    
    try:
        img_bytes = fig.to_image(format="png")
        st.download_button(
            label="Download AQI Trend Plot as PNG",
            data=img_bytes,
            file_name=f"aqi_trend_{city_select.lower()}.png",
            mime="image/png"
        )
    except Exception:
        st.info("Plotly image export requires kaleido. Install with: pip install -U kaleido")