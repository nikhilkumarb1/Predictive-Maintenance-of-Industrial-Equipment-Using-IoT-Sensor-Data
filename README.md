# Predictive Maintenance of Industrial Equipment using Machine Learning

## Project Overview
This project focuses on building a **Predictive Maintenance System** that analyzes industrial machine sensor data to detect abnormal patterns and predict potential machine failures. The goal is to reduce unexpected machine breakdowns, minimize downtime, and improve operational efficiency.

The system uses **Machine Learning (Anomaly Detection)** to identify unusual machine behavior and provides a **Streamlit dashboard** where users can upload machine sensor data and view machine health predictions.

---

## Problem Statement
Industrial machines generate large amounts of sensor data such as voltage, current, and power consumption. Unexpected machine failures can lead to:

- Production downtime
- Financial losses
- Safety risks

Traditional maintenance approaches such as reactive or scheduled maintenance are inefficient. This project implements a **predictive maintenance approach** that detects potential machine failures before they occur.

---

## Key Features
- Data preprocessing and cleaning
- Anomaly detection using **Isolation Forest**
- Machine health prediction
- Streamlit interactive dashboard
- Visualization of machine health status
- Downloadable prediction results

---

## Technology Stack

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

### Concepts Used
- Machine Learning
- Anomaly Detection
- Data Preprocessing
- Model Deployment

---

## Machine Learning Approach
The dataset used in this project contains mostly **healthy machine data** with very few failure cases. Because of this imbalance, an **Anomaly Detection approach** was used instead of traditional classification.

The **Isolation Forest algorithm** was trained using normal machine behavior to detect abnormal patterns that may indicate potential machine failure.

---

## Project Workflow
1. Data Collection  
2. Data Preprocessing  
3. Feature Engineering  
4. Model Training using Isolation Forest  
5. Model Evaluation  
6. Model Saving using Joblib  
7. Deployment using Streamlit Dashboard

---

## Dashboard Features
The Streamlit dashboard allows users to:

- Upload machine sensor datasets
- Run machine failure predictions
- View machine health status
- Analyze machine health summary
- Download prediction results

---

---

## How to Run the Project

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn streamlit joblib openpyxl
```
### 2. To run the app 
```bash
python -m streamlit run app.py
```
