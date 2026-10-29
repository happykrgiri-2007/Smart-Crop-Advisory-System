# 🌾 Module 3: Crop Profitability Analysis

## 📌 Overview
This module analyzes crop profitability using agricultural data, including crop support prices, cultivation costs, and yield information. It estimates revenue and profit per hectare for a selected crop and state.

## 🎯 Objectives
- Analyze crop-wise and state-wise agricultural data.
- Calculate estimated revenue per hectare.
- Analyze cultivation costs and crop profitability.
- Use Linear Regression to estimate production costs.
- Generate visualizations to understand profitability trends.

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn



## 📊 Dataset
The module uses `01_datafile.csv`, containing agricultural information used to analyze crop support prices, cultivation costs, yield, revenue, and profitability.

The exact dataset columns depend on the supplied CSV file and the column-renaming logic in the Python script.

## ⚙️ Methodology
1. Load the agricultural dataset.
2. Rename relevant columns into standardized names.
3. Prepare the data for analysis.
4. Calculate revenue per hectare.
5. Calculate estimated profit per hectare.
6. Apply Linear Regression for production-cost estimation.
7. Generate visualizations and profitability summaries.
8. Retrieve profitability estimates for a selected crop and state.

## 🚀 How to Run


```bash
python 02_Profitability_analysis.py
```

Run these commands from the Module 3 folder.

## 📈 Output
The module provides:
- Estimated revenue per hectare
- Estimated profit per hectare
- Crop and state profitability information
- Data analysis and visualizations

## 🔗 Integration
This module is part of the Smart Crop Advisory System and works alongside crop recommendation and crop yield prediction modules.

