# 📈 FAANG-tastic Insights

### **Predict FAANG Stock Prices using Machine Learning, MLflow & Streamlit**

This project builds an end-to-end **stock price prediction system for FAANG companies** (Facebook/Meta, Amazon, Apple, Netflix, Google).
It combines **data engineering, machine learning, model tracking (MLflow), and deployment through Streamlit** into one complete solution.

---

## 🚀 Project Overview

The goal is to build an intelligent, user-friendly **Streamlit web app** that predicts the **closing price** of FAANG stocks using regression models.
This tool empowers:

* **Investors** — see price predictions instantly.
* **Financial analysts** — analyze stock trends and compare ML models.
* **Traders** — get short-term insights into market movement.

---

## 🧠 Skills Demonstrated

✔ Data Cleaning & Preprocessing
✔ Exploratory Data Analysis (EDA)
✔ Feature Engineering
✔ Regression Modeling
✔ Hyperparameter Tuning
✔ MLflow Experiment Tracking
✔ Model Deployment
✔ Streamlit App Development
✔ Documentation & Reporting

---

## 🏢 Domain — Finance

This project focuses on **financial stock market data** and builds predictive insights for FAANG companies.

---

## 📌 Problem Statement

As a data scientist at a fintech company, your task is to develop a prediction system that forecasts the **closing price** of FAANG stocks from user inputs.
The system must be:

* Accurate 🟢
* Fast ⚡
* User-friendly 🖥
* Interpretable 🔍
* Trackable using MLflow

---

## 💼 Business Use Cases

| Use Case               | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| **Financial Advisory** | Predict future stock closings to guide investment decisions. |
| **Portfolio Analysis** | Understand and visualize stock performance trends.           |
| **Trading Strategy**   | Short-term forecasts for risk-aware trading.                 |

---

# 🧩 Project Workflow

## 1️⃣ Data Cleaning

* Handled missing values using mean/median/mode
* Converted `Date` into standard format
* Created `Year`, `Month`, `Day`
* Forward filled price columns
* Removed unnecessary financial metrics
* Clipped outliers using **IQR**
* Encoded company and ticker values
* Saved preprocessing artifacts:

  * `feature_columns.pkl`
  * `company_encoder.pkl`
  * `standard_scaler.pkl`

---

## 2️⃣ Exploratory Data Analysis (EDA)

Performed extensive visualizations:

* Close Price Over Time
* Volume Trends
* Yearly Average Close
* Volume vs Close Scatter
* Correlation Heatmap
* Boxplots of closing prices

These insights guided feature selection for the model.

---

## 3️⃣ Model Development

### Algorithms Used:

* Linear Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* SVR
* XGBoost

### Performance Metrics:

* **MAE**
* **RMSE**
* **R² Score**

**Best model:** 🥇 *Linear Regression*

* R² Score = **1.0000**
* MAE = **0.41**
* RMSE = **0.58**

All models were tracked using **MLflow (DagsHub Integration)**.

---

## 4️⃣ MLflow Integration

Connected MLflow to DagsHub:

✔ Logged hyperparameters
✔ Logged metrics (MAE, RMSE, R²)
✔ Uploaded trained models as artifacts
✔ Best model identified and stored

**MLflow Dashboard:**
You can view all runs here:
👉 [https://dagshub.com/tstr12cg429/my-first-repo.mlflow/#/experiments/0](https://dagshub.com/tstr12cg429/my-first-repo.mlflow/#/experiments/0)

---

## 5️⃣ Model Deployment — Streamlit

A fully interactive dashboard:

### **Features:**

* Sidebar inputs for model parameters
* Prediction tab
* Historical data tab (Yahoo Finance API)
* Model Comparison table + chart
* Explainability tab using **SHAP**
* Downloadable prediction vectors
* Artifact status indicators
* MLflow run links embedded

The app loads:

```
best_model.pkl
feature_columns.pkl
company_encoder.pkl
standard_scaler.pkl
```

---

## 6️⃣ Model Explainability (SHAP)

* Beeswarm plot
* Bar plot (global importance)
* Waterfall explanation for first record

Users can upload CSV files to visualize feature contributions.

---

## 🗂 Dataset

**Name:** FAANG Financial Dataset
**Format:** CSV
**Rows:** 23,055
**Companies:** Apple, Amazon, Google, Netflix, Meta

### **Key Columns Used:**

| Column           | Description              |
| ---------------- | ------------------------ |
| Open             | Opening price            |
| High             | Highest price            |
| Low              | Lowest price             |
| Close            | Closing price (target)   |
| Volume           | Shares traded            |
| Market Cap       | Company market valuation |
| PE Ratio         | Valuation metric         |
| EPS              | Earnings per share       |
| ROE              | Return on Equity         |
| Debt to Equity   | Leverage indicator       |
| Profit Margin    | Profitability            |
| Enterprise Value | Fair value indicator     |

Features were transformed into:

✔ Normalized values
✔ Encoded company/tickers
✔ One-hot vectors
✔ Date features

---

# 📊 Final Results

### **💡 Best Model:** Linear Regression

| Metric | Value  |
| ------ | ------ |
| MAE    | 0.4121 |
| RMSE   | 0.5830 |
| R²     | 1.0000 |

The model demonstrated highly accurate prediction capabilities on the test set.

---

# 🖥️ Streamlit App Structure

```
📁 Project
│── app.py
│── best_model.pkl
│── standard_scaler.pkl
│── feature_columns.pkl
│── company_encoder.pkl
│── Model_Comparison.png
│── README.md
```

Run the app:

```
streamlit run app.py
```

---

# 📦 Deliverables

✔ Complete Source Code
✔ MLflow Tracking Dashboard
✔ Streamlit Web App
✔ Data Preprocessing Scripts
✔ Trained Model Files
✔ EDA Visualizations
✔ Detailed Documentation
✔ Project Presentation Slides

---

# 🧭 Project Timeline

**Completion Time: 1 Week**

---

# ✨ Author

**Sree V G**
FAANG Stock Prediction — ML · Streamlit · MLflow

