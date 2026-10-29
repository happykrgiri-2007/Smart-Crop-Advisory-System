# Crop Recommendation System 🌱

## Overview
The Crop Recommendation System is a Machine Learning project
that recommends a suitable crop based on soil nutrients and
weather conditions.

The system uses Nitrogen (N), Phosphorus (P), Potassium (K),
temperature, humidity, soil pH, and rainfall to predict the
most suitable crop.

## Objectives
- Analyze the crop recommendation dataset.
- Perform Exploratory Data Analysis (EDA).
- Train Decision Tree and Random Forest classifiers.
- Evaluate model performance using accuracy and classification reports.
- Identify important features using Random Forest.
- Recommend a suitable crop based on input values.

## Dataset
Dataset file: `Crop_recommendation.csv`

The dataset contains 2,200 records and 8 columns.

### Features
- N: Nitrogen content
- P: Phosphorus content
- K: Potassium content
- temperature: Temperature in Celsius
- humidity: Relative humidity
- ph: Soil pH value
- rainfall: Rainfall value

### Target
- label: Recommended crop name

## Machine Learning Algorithms
1. Decision Tree Classifier
2. Random Forest Classifier

## Technologies Used
- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure

Crop_Recommendation_System/
│
├── Crop_recommendation.csv
├── Crop_Recommendation.py
├── README.md
└── requirements.txt

