# Crop Recommendation System

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "01_Crop_recommendation.csv")
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCrop Distribution:")
print(df["label"].value_counts())

# 2. Data visualization
sns.countplot(data=df, x="label")
plt.xticks(rotation=90)
plt.title("Crop Distribution")
plt.tight_layout()
plt.show()

# 3. Features and target
X = df[["N", "P", "K", "temperature",
        "humidity", "ph", "rainfall"]]

y = df["label"]

# 4. Encode crop labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# 5. Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Decision Tree Model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\nDecision Tree Accuracy:")
print(accuracy_score(y_test, dt_pred) * 100, "%")

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_pred))

# 7. Random Forest Model
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRandom Forest Accuracy:")
print(accuracy_score(y_test, rf_pred) * 100, "%")

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# 8. Feature Importance
importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

importance.plot(kind="bar")
plt.title("Random Forest Feature Importance")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# 9. Crop Recommendation Function
def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):

    data = pd.DataFrame([[
        N, P, K, temperature, humidity, ph, rainfall
    ]], columns=[
        "N", "P", "K", "temperature",
        "humidity", "ph", "rainfall"
    ])

    prediction = rf.predict(data)

    crop = encoder.inverse_transform(prediction)[0]

    return crop

# 10. Test Recommendation
crop = recommend_crop(
    90, 42, 43, 20.88, 82.00, 6.50, 202.94
)

print("\nRecommended Crop:", crop)

print("\nCrop Recommendation System Completed!")