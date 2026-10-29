
# ============================================================
# MODULE 3: CROP PROFITABILITY ANALYSIS
# Smart Crop Advisory System
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# STEP 1: CREATE OUTPUT FOLDER
# ------------------------------------------------------------

os.makedirs("outputs", exist_ok=True)

# ------------------------------------------------------------
# STEP 2: LOAD DATASET
# ------------------------------------------------------------

print("\n========== CROP PROFITABILITY ANALYSIS ==========\n")

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "01_datafile.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

# ------------------------------------------------------------
# STEP 3: RENAME COLUMNS
# ------------------------------------------------------------

df = df.rename(columns={
    "Cost of Cultivation (`/Hectare) A2+FL":
        "Cost_Cultivation_A2FL",

    "Cost of Cultivation (`/Hectare) C2":
        "Cost_Cultivation_C2",

    "Cost of Production (`/Quintal) C2":
        "Cost_Production_C2",

    "Yield (Quintal/ Hectare) ":
        "Yield_Quintal_Hectare",

    "Support price":
        "Support_Price"
})

print("\nRenamed columns:")
print(df.columns.tolist())

# ------------------------------------------------------------
# STEP 4: DATA CLEANING
# ------------------------------------------------------------

# Remove extra spaces from crop and state names
df["Crop"] = df["Crop"].astype(str).str.strip().str.upper()
df["State"] = df["State"].astype(str).str.strip()

numeric_columns = [
    "Cost_Cultivation_A2FL",
    "Cost_Cultivation_C2",
    "Cost_Production_C2",
    "Yield_Quintal_Hectare",
    "Support_Price"
]

# Convert numerical columns to numeric values
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove missing and infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=["Crop", "State"] + numeric_columns, inplace=True)

# Remove invalid negative values and zero yield
df = df[
    (df["Cost_Cultivation_A2FL"] >= 0) &
    (df["Cost_Cultivation_C2"] >= 0) &
    (df["Cost_Production_C2"] >= 0) &
    (df["Yield_Quintal_Hectare"] > 0) &
    (df["Support_Price"] >= 0)
].copy()

print("\nDataset shape after cleaning:", df.shape)

# ------------------------------------------------------------
# STEP 5: CALCULATE REVENUE AND ESTIMATED PROFIT
# ------------------------------------------------------------

# Estimated revenue per hectare using support price
df["Revenue_per_Hectare"] = (
    df["Support_Price"] * df["Yield_Quintal_Hectare"]
)

# Estimated profit per hectare using C2 cultivation cost
df["Profit_per_Hectare"] = (
    df["Revenue_per_Hectare"] - df["Cost_Cultivation_C2"]
)

print("\nRevenue and profit calculated successfully!")

print("\nCrop-wise data:")
print(
    df[
        [
            "Crop",
            "State",
            "Yield_Quintal_Hectare",
            "Support_Price",
            "Revenue_per_Hectare",
            "Profit_per_Hectare"
        ]
    ].head(10).round(2)
)

# Save cleaned data
df.to_csv("outputs/cleaned_profitability_data.csv", index=False)

# ------------------------------------------------------------
# STEP 6: CROP-WISE PROFITABILITY ANALYSIS
# ------------------------------------------------------------

crop_profit = (
    df.groupby("Crop")["Profit_per_Hectare"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== AVERAGE PROFIT BY CROP ==========")
print(crop_profit.round(2))

# Save crop-wise profit results
crop_profit.reset_index().to_csv(
    "outputs/crop_wise_profit.csv",
    index=False
)

# ------------------------------------------------------------
# STEP 7: VISUALIZE CROP PROFITABILITY
# ------------------------------------------------------------

plt.figure(figsize=(12, 7))

sns.barplot(
    x=crop_profit.values,
    y=crop_profit.index
)

plt.title("Average Estimated Profit per Hectare by Crop")
plt.xlabel("Estimated Profit per Hectare")
plt.ylabel("Crop")
plt.tight_layout()

plt.savefig(
    "outputs/crop_profitability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# STEP 8: FIND HIGHEST AND LOWEST AVERAGE PROFIT
# ------------------------------------------------------------

if not crop_profit.empty:
    highest_crop = crop_profit.idxmax()
    lowest_crop = crop_profit.idxmin()

    print("\n========== KEY FINDINGS ==========")
    print(
        f"Highest average estimated profit: {highest_crop} "
        f"({crop_profit.max():.2f} per hectare)"
    )
    print(
        f"Lowest average estimated profit: {lowest_crop} "
        f"({crop_profit.min():.2f} per hectare)"
    )

# ------------------------------------------------------------
# STEP 9: LINEAR REGRESSION MODEL
# Predict production cost per quintal
# ------------------------------------------------------------

features = [
    "Cost_Cultivation_A2FL",
    "Cost_Cultivation_C2",
    "Yield_Quintal_Hectare"
]

target = "Cost_Production_C2"

X = df[features]
y = df[target]

model = LinearRegression()
model.fit(X, y)

# This is training-set evaluation, not unseen test performance
predictions = model.predict(X)

r2 = r2_score(y, predictions)
mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))

print("\n========== LINEAR REGRESSION RESULTS ==========")
print("Model: Linear Regression")
print(f"Training R2 Score: {r2:.4f}")
print(f"Training MAE: {mae:.2f}")
print(f"Training RMSE: {rmse:.2f}")

print(
    "\nNote: These metrics are calculated on the same data used "
    "for training. They do not show how accurately the model "
    "will predict costs for new data."
)

# ------------------------------------------------------------
# STEP 10: ACTUAL VS PREDICTED COST GRAPH
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(y, predictions)

min_value = min(y.min(), predictions.min())
max_value = max(y.max(), predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title("Actual vs Predicted Production Cost")
plt.xlabel("Actual Production Cost per Quintal")
plt.ylabel("Predicted Production Cost per Quintal")
plt.tight_layout()

plt.savefig(
    "outputs/actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# STEP 11: CROP AND STATE PROFIT ESTIMATE FUNCTION
# ------------------------------------------------------------

def get_profit_estimate(crop, state):
    """
    Return a historical dataset-based estimate for a crop and state.
    Returns None if the crop-state combination is unavailable.
    """

    crop = str(crop).strip().upper()
    state = str(state).strip()

    result = df[
        (df["Crop"].str.upper() == crop) &
        (df["State"].str.casefold() == state.casefold())
    ]

    if result.empty:
        return None

    row = result.iloc[0]

    return {
        "Crop": row["Crop"],
        "State": row["State"],
        "Support_Price": round(row["Support_Price"], 2),
        "Cultivation_Cost_C2": round(
            row["Cost_Cultivation_C2"], 2
        ),
        "Yield_Quintal_Hectare": round(
            row["Yield_Quintal_Hectare"], 2
        ),
        "Estimated_Revenue_per_Hectare": round(
            row["Revenue_per_Hectare"], 2
        ),
        "Estimated_Profit_per_Hectare": round(
            row["Profit_per_Hectare"], 2
        )
    }


# ------------------------------------------------------------
# STEP 12: TEST THE PROFIT ESTIMATE FUNCTION
# ------------------------------------------------------------

print("\n========== CROP-STATE PROFIT ESTIMATE ==========")

# Example 1
result = get_profit_estimate("MAIZE", "Bihar")

if result is not None:
    for key, value in result.items():
        print(f"{key}: {value}")
else:
    print("Crop-state combination not found.")

# Example 2: test an unavailable combination
result = get_profit_estimate("UNKNOWN CROP", "Unknown State")

if result is None:
    print("\nUnknown crop-state combination was handled correctly.")

# ------------------------------------------------------------
# STEP 13: SAVE FINAL RESULTS
# ------------------------------------------------------------

df.to_csv("outputs/profitability_results.csv", index=False)

print("\nAll analysis steps completed successfully!")
print("Check the 'outputs' folder for CSV files and graphs.")
