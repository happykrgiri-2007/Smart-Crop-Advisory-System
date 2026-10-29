
# ============================================================
# MODULE 2: CROP YIELD PREDICTION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# STEP 1: LOAD DATASET
# ------------------------------------------------------------


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "01_Crop Prediction dataset.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# STEP 2: DATA CLEANING
# ------------------------------------------------------------

print("\nSTEP 2: Cleaning Dataset")

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip()

# Remove extra spaces from text columns
categorical_columns = [
    "State_Name", "District_Name", "Season", "Crop"
]

for column in categorical_columns:
    df[column] = df[column].astype("string").str.strip()

# Convert numerical columns to numeric values
numerical_columns = [
    "Crop_Year", "Temperature", "Humidity",
    "Soil_Moisture", "Area", "Production"
]

for column in numerical_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Replace infinite values with missing values
df = df.replace([np.inf, -np.inf], np.nan)

# Remove rows with missing required values
df = df.dropna(
    subset=categorical_columns + numerical_columns
)

# Remove rows where area or production is zero or negative
df = df[(df["Area"] > 0) & (df["Production"] > 0)]

print("Shape After Cleaning:", df.shape)
print("Remaining Missing Values:", df.isnull().sum().sum())


# ------------------------------------------------------------
# STEP 3: FEATURE ENGINEERING
# ------------------------------------------------------------

print("\nSTEP 3: Calculating Crop Yield")

# Yield = Production / Area
df["Yield"] = df["Production"] / df["Area"]

# Remove invalid yield values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["Yield"])
df = df[df["Yield"] > 0]

# Remove extreme yield outliers above the 95th percentile
yield_limit = df["Yield"].quantile(0.95)
df = df[df["Yield"] <= yield_limit].copy()

print("95th Percentile Yield Limit:", yield_limit)
print("Shape After Outlier Removal:", df.shape)

print("\nYield Statistics:")
print(df["Yield"].describe())


# ------------------------------------------------------------
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------

print("\nSTEP 4: Exploratory Data Analysis")

print("\nTop 10 Crops by Average Yield:")
print(
    df.groupby("Crop")["Yield"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

# Yield distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Yield"], bins=40, kde=True)
plt.title("Distribution of Crop Yield")
plt.xlabel("Yield (Production / Area)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Average yield by crop
top_crops = (
    df.groupby("Crop")["Yield"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
top_crops.sort_values().plot(kind="barh")
plt.title("Top 10 Crops by Average Yield")
plt.xlabel("Average Yield")
plt.ylabel("Crop")
plt.tight_layout()
plt.show()

# Correlation heatmap for numerical columns
plt.figure(figsize=(9, 6))
sns.heatmap(
    df[
        ["Crop_Year", "Temperature", "Humidity",
         "Soil_Moisture", "Area", "Production", "Yield"]
    ].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# STEP 5: ENCODE CATEGORICAL FEATURES
# ------------------------------------------------------------

print("\nSTEP 5: Encoding Categorical Features")

# Keep original dataset text for the prediction function
encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column + "_Encoded"] = encoder.fit_transform(df[column])
    encoders[column] = encoder

print("Categorical Encoding Completed")


# ------------------------------------------------------------
# STEP 6: SELECT FEATURES AND TARGET
# ------------------------------------------------------------

print("\nSTEP 6: Selecting Features")

feature_columns = [
    "State_Name_Encoded",
    "District_Name_Encoded",
    "Crop_Year",
    "Season_Encoded",
    "Crop_Encoded",
    "Temperature",
    "Humidity",
    "Soil_Moisture"
]

X = df[feature_columns]
y = df["Yield"]

print("Features:", feature_columns)
print("Target: Yield")
print("X Shape:", X.shape)
print("y Shape:", y.shape)


# ------------------------------------------------------------
# STEP 7: TRAIN-TEST SPLIT
# ------------------------------------------------------------

print("\nSTEP 7: Splitting Dataset")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


# ------------------------------------------------------------
# STEP 8: TRAIN LINEAR REGRESSION MODEL
# ------------------------------------------------------------

print("\nSTEP 8: Training Linear Regression")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_r2 = r2_score(y_test, lr_predictions)
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))

print("\nLinear Regression Results")
print("R2 Score:", round(lr_r2, 4))
print("MAE:", round(lr_mae, 4))
print("RMSE:", round(lr_rmse, 4))


# ------------------------------------------------------------
# STEP 9: TRAIN RANDOM FOREST REGRESSOR
# ------------------------------------------------------------

print("\nSTEP 9: Training Random Forest Regressor")

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_r2 = r2_score(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

print("\nRandom Forest Results")
print("R2 Score:", round(rf_r2, 4))
print("MAE:", round(rf_mae, 4))
print("RMSE:", round(rf_rmse, 4))


# ------------------------------------------------------------
# STEP 10: COMPARE BOTH MODELS
# ------------------------------------------------------------

print("\nSTEP 10: Model Comparison")

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R2 Score": [lr_r2, rf_r2],
    "MAE": [lr_mae, rf_mae],
    "RMSE": [lr_rmse, rf_rmse]
})

print(results.to_string(index=False))

# Higher R2 is better; lower MAE and RMSE are better
best_model = rf_model if rf_r2 >= lr_r2 else lr_model

print("\nBest Model Based on R2 Score:")
print("Random Forest" if best_model is rf_model
      else "Linear Regression")


# ------------------------------------------------------------
# STEP 11: ACTUAL VS PREDICTED YIELD
# ------------------------------------------------------------

print("\nSTEP 11: Actual vs Predicted Yield")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_predictions, alpha=0.5)

minimum = min(y_test.min(), rf_predictions.min())
maximum = max(y_test.max(), rf_predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title("Actual vs Predicted Crop Yield - Random Forest")
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# STEP 12: FEATURE IMPORTANCE
# ------------------------------------------------------------

print("\nSTEP 12: Feature Importance")

importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance", ascending=False
)

print(importance.to_string(index=False))

plt.figure(figsize=(9, 6))
sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# STEP 13: CROP YIELD PREDICTION FUNCTION
# ------------------------------------------------------------

def predict_yield(
    state,
    district,
    season,
    crop,
    year,
    temperature,
    humidity,
    soil_moisture
):
    """
    Predict crop yield using the better-performing model.

    Yield is predicted as Production / Area, in the dataset's
    corresponding yield units.
    """

    values = {
        "State_Name": state.strip(),
        "District_Name": district.strip(),
        "Season": season.strip(),
        "Crop": crop.strip()
    }

    encoded_values = {}

    for column, value in values.items():
        encoder = encoders[column]

        if value not in encoder.classes_:
            raise ValueError(
                f"Unknown {column}: '{value}'. "
                f"Use a value found in the dataset."
            )

        encoded_values[column + "_Encoded"] = int(
            encoder.transform([value])[0]
        )

    input_data = pd.DataFrame([{
        "State_Name_Encoded": encoded_values["State_Name_Encoded"],
        "District_Name_Encoded": encoded_values["District_Name_Encoded"],
        "Crop_Year": int(year),
        "Season_Encoded": encoded_values["Season_Encoded"],
        "Crop_Encoded": encoded_values["Crop_Encoded"],
        "Temperature": float(temperature),
        "Humidity": float(humidity),
        "Soil_Moisture": float(soil_moisture)
    }], columns=feature_columns)

    prediction = best_model.predict(input_data)[0]

    return max(0.0, float(prediction))


# ------------------------------------------------------------
# STEP 14: TEST THE PREDICTION FUNCTION
# ------------------------------------------------------------

print("\nSTEP 14: Testing Prediction Function")

# Use an actual row from the cleaned dataset so that all
# categorical values are valid.
sample = df.iloc[0]

predicted_yield = predict_yield(
    state=str(sample["State_Name"]),
    district=str(sample["District_Name"]),
    season=str(sample["Season"]),
    crop=str(sample["Crop"]),
    year=int(sample["Crop_Year"]),
    temperature=float(sample["Temperature"]),
    humidity=float(sample["Humidity"]),
    soil_moisture=float(sample["Soil_Moisture"])
)

print("\nSample Input:")
print("State:", sample["State_Name"])
print("District:", sample["District_Name"])
print("Season:", sample["Season"])
print("Crop:", sample["Crop"])
print("Year:", sample["Crop_Year"])
print("Temperature:", sample["Temperature"])
print("Humidity:", sample["Humidity"])
print("Soil Moisture:", sample["Soil_Moisture"])

print("\nPredicted Crop Yield:", round(predicted_yield, 4))
print("Actual Crop Yield:", round(float(sample["Yield"]), 4))

print("\nMODULE 2 COMPLETED SUCCESSFULLY!")
