# Household Energy Consumption Prediction

## Problem Statement
In the modern world, energy management is a critical issue for both households and energy providers. Predicting energy consumption accurately enables better planning, cost reduction, and optimization of resources. The goal of this project is to develop a machine learning model that can predict household energy consumption based on historical data. Using this model, consumers can gain insights into their usage patterns, while energy providers can forecast demand more effectively.

By the end of this project, learners should provide actionable insights into energy usage trends and deliver a predictive model that can help optimize energy consumption for households or serve as a baseline for further research into energy management systems.

---

##  Repository Contents
This repository contains:
- `household_power.py` – Python script with full machine learning pipeline
- `app.py` - visual approch of the data
- `individual+household+electric+power+consumption.zip` – Zipped dataset file (must be extracted to access `.txt` file)

---

## Required Packages
This project uses the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `streamlit`
- `scikit-learn`

###  Install Packages with pip
```bash
pip install pandas numpy matplotlib seaborn streamlit scikit-learn
```

---

## 🚀 How to Run the App
1. Run ipynb file using play Button in vs code
2. Run the Streamlit application with the command:
```bash
app.py
```

---

## Project Workflow (Step-by-Step)
1. **Load Dataset** – Read the household energy consumption data
2. **Clean Data** – Remove missing values and invalid records
3. **Outlier Handling** – Use IQR capping to limit extreme values
4. **Correlation Analysis** – Identify relationships between features
7. **Feature Selection & Splitting** – Select inputs and divide into training and testing sets
8. **Model Training & Evaluation**:
   - Linear Regression
   - KNN Regression
   - Decision Tree Regression
   - Random Forest Regression
   - Ridge Regression
9. **Visualization** – Display R², MAE, MSE, RMSE scores in Streamlit dashboard

---
