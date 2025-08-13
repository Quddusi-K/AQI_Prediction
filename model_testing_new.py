import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib
import glob

# Load feature data
data_file = max(glob.glob("data/historical_combined_cities_*.csv"), key=os.path.getctime)
df = pd.read_csv(data_file, parse_dates=["time"])
# Drop all aqi_lag features
df.drop(columns=[col for col in df.columns if "aqi_lag" in col], inplace=True)

df.set_index("time", inplace=True)

# Define city info
city_info = [
    {"name": "Karachi", "city_code": 0},
    {"name": "Islamabad", "city_code": 1},
    {"name": "Lahore", "city_code": 2},
]

# Define model configurations (for initializing new models)
model_configs = {
    "Ridge": lambda: Ridge(alpha=1.0),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=10, random_state=42),
    "GradientBoosting": lambda: GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=5,
                                                         subsample=0.8, random_state=42, validation_fraction=0.1,
                                                         n_iter_no_change=10),
    "XGBoost": lambda: XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
}

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Train and evaluate per city
results = []
predictions = {}
for city in city_info:
    city_df = df[df["city"] == city["city_code"]]
    split_idx = int(len(city_df) * 0.8)
    train = city_df.iloc[:split_idx]
    test = city_df.iloc[split_idx:]
    print(f"\n{city['name']} - Training set size: {len(train)}, Test set size: {len(test)} \n")
    
    X_train = train.drop(columns=["us_aqi"])
    y_train = train["us_aqi"]
    X_test = test.drop(columns=["us_aqi"])
    y_test = test["us_aqi"]
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    for name in model_configs.keys():
        # Load existing model or initialize new one
        model_file = f"model/{name}_{city['name']}.joblib"
        if os.path.exists(model_file):
            print(f"Loading saved {name} for {city['name']}...")
            model = joblib.load(model_file)
        else:
            print(f"Initializing new {name} for {city['name']}...")
            model = model_configs[name]()
        
        # Load or initialize scaler for Ridge
        scaler_file = f"model/scaler_{city['name']}.joblib"
        if name == "Ridge":
            if os.path.exists(scaler_file):
                print(f"Loading saved scaler for {city['name']}...")
                scaler = joblib.load(scaler_file)
            else:
                print(f"Initializing new scaler for {city['name']}...")
                scaler = StandardScaler()
        else:
            scaler = None
        
        print(f"Training {name} for {city['name']}...")
        cv_scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            if name == "Ridge":
                X_tr_scaled = scaler.fit_transform(X_tr)
                X_val_scaled = scaler.transform(X_val)
            else:
                X_tr_scaled, X_val_scaled = X_tr, X_val
            
            model.fit(X_tr_scaled, y_tr)
            y_pred_cv = model.predict(X_val_scaled)
            cv_scores.append(root_mean_squared_error(y_val, y_pred_cv))
        cv_rmse = np.mean(cv_scores)
        
        # Train on full training set
        if name == "Ridge":
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        predictions[(city["name"], name)] = y_pred
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({
            "City": city["name"],
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "CV_RMSE": cv_rmse
        })
        
        # Save model and scaler
        joblib.dump(model, model_file)
        if name == "Ridge":
            joblib.dump(scaler, scaler_file)
        print(f"Saved {name} model for {city['name']}.\n")

# Show results
results_df = pd.DataFrame(results)
print("\nModel Comparison by City:")
print(results_df.sort_values(by=["City", "RMSE"]))

# Uncomment the following lines to visualize predictions and feature importance
# Visualize predictions
# for city in city_info:
#     plt.figure(figsize=(12, 6))
#     city_test = df[df["city"] == city["city_code"]].iloc[int(len(df[df["city"] == city["city_code"]]) * 0.8):]
#     y_test = city_test["us_aqi"]
#     plt.plot(y_test.index, y_test.values, label="Actual", color="black", linewidth=2)
#     colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
#     for idx, name in enumerate(models.keys()):
#         plt.plot(y_test.index, predictions[(city["name"], name)], label=name, color=colors[idx])
#     plt.legend()
#     plt.title(f"AQI Prediction Comparison for {city['name']}")
#     plt.xlabel("Time")
#     plt.ylabel("US AQI")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()


# Feature importance for XGBoost
# for city in city_info:
#     model = joblib.load(f"model/XGBoost_{city['name']}.joblib")
#     # Get feature importance by gain
#     importance_dict = model.get_booster().get_score(importance_type="gain")
#     importance = pd.DataFrame({
#         "Feature": list(importance_dict.keys()),
#         "Importance": list(importance_dict.values())
#     })
#     importance = importance.sort_values(by="Importance", ascending=False)
#     plt.figure(figsize=(10, 6))
#     plt.barh(importance["Feature"], importance["Importance"])
#     plt.title(f"Feature Importance (Gain) for XGBoost ({city['name']})")
#     plt.xlabel("Gain")
#     plt.tight_layout()
#     plt.show()

# # Residual analysis
# for city in city_info:
#     plt.figure(figsize=(10, 6))
#     city_test = df[df["city"] == city["city_code"]].iloc[int(len(df[df["city"] == city["city_code"]]) * 0.8):]
#     y_test = city_test["us_aqi"]
#     residuals = y_test - predictions[(city["name"], "XGBoost")]
#     plt.scatter(y_test, residuals, alpha=0.5)
#     plt.axhline(0, color="red", linestyle="--")
#     plt.xlabel("Actual AQI")
#     plt.ylabel("Residuals (Actual - Predicted)")
#     plt.title(f"Residuals vs Actual AQI (XGBoost) for {city['name']}")
#     plt.tight_layout()
#     plt.show()