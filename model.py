# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
# import joblib
# import pandas as pd
# import os

# df = pd.read_csv("historical_features.csv", parse_dates=["time"])
# df.set_index("time", inplace=True)

# split_idx = int(len(df) * 0.8)
# train = df.iloc[:split_idx]
# test = df.iloc[split_idx:]

# X_train = train.drop(columns=["us_aqi"])
# y_train = train["us_aqi"]
# X_test = test.drop(columns=["us_aqi"])
# y_test = test["us_aqi"]


# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Predict
# y_pred = model.predict(X_test)

# # Evaluate
# print("MAE:", mean_absolute_error(y_test, y_pred))
# print("RMSE:", root_mean_squared_error(y_test, y_pred))
# print("R² Score:", r2_score(y_test, y_pred))

# # Save the model
# joblib.dump(model, "model/aqi_rf_model.joblib")
# os.makedirs("model", exist_ok=True)

import pandas as pd
import os
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load feature data
df = pd.read_csv("data/historical_combined.csv", parse_dates=["time"])
df.set_index("time", inplace=True)

# Split chronologically
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]
# print len of train and test sets
print(f"Training set size: {len(train)}, Test set size: {len(test)}")

X_train = train.drop(columns=["us_aqi"])
y_train = train["us_aqi"]
X_test = test.drop(columns=["us_aqi"])
y_test = test["us_aqi"]

# Define models to compare
models = {
    "LinearRegression": LinearRegression(),
    "RidgeRegression": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(
        n_estimators=100,      # 100-200 trees is reasonable for 2000 samples
        max_depth=10,          # limit depth to prevent overfitting
        min_samples_split=10,  # require at least 10 samples to split a node
        random_state=42
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100,      # 100 boosting stages
        learning_rate=0.1,     # default, can tune lower for more trees
        max_depth=5,           # shallower trees for boosting
        subsample=0.8,         # use 80% of data for each tree to reduce overfitting
        random_state=42
    )
}

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Results
results = []

predictions = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    # Save model
    import joblib
    joblib.dump(model, f"model/{name}.joblib")

# Visualize all predictions together
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual", color="black", linewidth=2)
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
for idx, (name, y_pred) in enumerate(predictions.items()):
    plt.plot(y_pred, label=name, color=colors[idx % len(colors)])
plt.legend()
plt.title("AQI Prediction Comparison Across Models")
plt.xlabel("Sample Index")
plt.ylabel("US AQI")
plt.tight_layout()
plt.show()

# Show results
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.sort_values(by="RMSE"))
