# 📉 Customer Churn Prediction System

A **machine learning–powered churn prediction system** developed during a hands-on internship project.  
The system identifies customers who are at risk of leaving a service and presents actionable insights through an **interactive Streamlit application** that business stakeholders can easily use.

---

## 🔍 Project Overview

Customer churn has a direct impact on revenue in industries such as telecom, SaaS, and banking.  
This project delivers a **complete end-to-end machine learning solution**, from raw data ingestion to deployment, enabling businesses to **proactively manage customer retention**.

Key objectives of the project:
- Predict churn probability for individual customers
- Compare performance between **Logistic Regression** and **Random Forest**
- Identify and visualize the main drivers of churn
- Deploy a user-friendly interface for non-technical stakeholders via Streamlit

---

## 📊 Dataset

- **Source**: Telco Customer Churn dataset (Kaggle)  
- **Target variable**: `Churn` (Yes/No)
- **Feature types**:
  - **Categorical**: Contract type, payment method, services subscribed, etc.
  - **Numerical**: Tenure, monthly charges, total charges

> The dataset is accessed dynamically via the Kaggle API during development and is **not stored as a CSV in this repository**.

---

## 🛠️ Technology Stack

- **Programming Language**: Python
- **Data Manipulation**: pandas, numpy
- **Machine Learning & Evaluation**: scikit-learn, imbalanced-learn (SMOTE)
- **Visualization**: matplotlib
- **Web Application**: Streamlit
- **Model Serialization**: pickle

---

## ⚙️ Pipeline Architecture

The system follows a **robust, modular ML pipeline**:

1. **Data Cleaning & Preprocessing**  
   Handling missing values, correcting data types, and preparing features for modeling.

2. **Encoding & Scaling**  
   - OneHotEncoding for categorical variables  
   - Standard scaling for numerical variables

3. **Class Imbalance Handling**  
   - Addressing skewed target distribution using **SMOTE**

4. **Model Training & Evaluation**  
   - Train multiple models (Logistic Regression & Random Forest)  
   - Evaluate using comprehensive metrics

5. **Serialization**  
   - Save trained models and preprocessing pipeline using pickle

6. **Deployment**  
   - Interactive web application via Streamlit

---

## 🤖 Models Used

### Logistic Regression
- Serves as a **baseline and interpretable model**
- Strong **ROC-AUC performance**
- Coefficients used to interpret churn drivers

### Random Forest
- **Ensemble-based, non-linear model**  
- Captures complex feature interactions
- Provides **feature importance** for deeper insight

---

## 📈 Evaluation Metrics

Model performance is evaluated using multiple metrics to ensure reliable predictions:

- **Accuracy** – overall correctness of predictions
- **Precision** – correctness of positive churn predictions
- **Recall** – ability to identify all actual churners
- **F1-score** – harmonic mean of precision and recall
- **ROC-AUC** – discriminatory power of the model
- **Confusion Matrix** – detailed class-wise prediction summary
- **ROC Curve** – visual representation of model performance

A **model comparison analysis** justifies the selection of the final model.

---

## 🖥️ Streamlit Application

The **interactive Streamlit app** allows business users to:

- Input customer-specific details  
- Select a trained model for prediction  
- View churn probability for individual customers  
- Compare predictions across multiple models

Deployed App: [Customer Churn Prediction App](https://ananyascodehq-customer-churn-prediction-app.streamlit.app/)


TRYMORE MHLANGA