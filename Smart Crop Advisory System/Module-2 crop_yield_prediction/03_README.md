# 🌾 Module 2: Crop Yield Prediction

## 📌 Overview
This module predicts agricultural crop yield using machine learning and historical agricultural data. It analyzes crop, location, season, weather, soil moisture, cultivated area, and production information to estimate crop yield.

## 🎯 Objectives
- Analyze historical crop production data.
- Clean and preprocess agricultural data.
- Calculate crop yield using production and cultivated area.
- Train and evaluate machine learning models.
- Predict crop yield for selected agricultural conditions.

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## 📂 Project Structure

```text
Module-2 crop_yield_prediction/
├── 01_Crop Prediction dataset.csv
├── 02_crop_yield_prediction.py
├── 03_README.md
├── 04_requirement.txt
├── 05_crop_yield_distribution.png
├── 06_crop_yield_average.png
├── 07_correlation_heatmap.png
├── 08_Actual vs predicted.png
└── 09_.gitignore
```

*Note: The visualization filenames above are based on the project structure and may need adjustment to match your actual files.*

## 📊 Dataset
The dataset contains agricultural records with fields such as:
- State and district
- Crop year and season
- Crop name
- Temperature and humidity
- Soil moisture
- Cultivated area
- Production

The dataset file expected by the Python script is `01_Crop Prediction dataset.csv`.

## ⚙️ Methodology
1. Load the agricultural dataset.
2. Inspect missing values and data types.
3. Clean and preprocess the data.
4. Calculate yield from production and area.
5. Handle extreme yield values.
6. Encode categorical features.
7. Train and evaluate regression models.
8. Use the selected model to predict crop yield.
9. Generate visualizations for analysis.

## 🚀 How to Run



```bash
python 02_crop_yield_prediction.py


Run these commands from the Module 2 folder.

## 📈 Output
The module provides:
- Crop yield predictions
- Model evaluation results
- Yield distribution visualizations
- Actual-versus-predicted comparisons
- Agricultural data analysis

## 🔗 Integration
This module is integrated into the Smart Crop Advisory System alongside crop recommendation and crop profitability analysis.

