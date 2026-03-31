# Predictive Maintenance of Industrial Equipment Using IoT Sensor Data
## Final Year Project Report & Interview Preparation Guide

---

### 1. Problem Statement
Unexpected machine failures in industrial environments lead to costly downtime, expensive repairs, and potential safety hazards. Traditional reactive maintenance (fixing after failure) or preventative maintenance (fixing on a schedule) are inefficient. This project aims to predict machine failures before they occur by analyzing continuous streams of IoT sensor data, enabling *predictive maintenance* to optimize operational efficiency and reduce costs.

### 2. Objectives
- Generate a realistic synthetic dataset mimicking industrial IoT sensors (temperature, vibration, pressure, current, etc.).
- Perform Exploratory Data Analysis (EDA) to find correlations between sensor spikes and machine failures.
- Train and compare Machine Learning classification models to predict machine health (Normal vs. Failure).
- Deploy an interactive Streamlit dashboard allowing users to input real-time sensor data and receive maintenance recommendations.

### 3. Methodology
1. **Data Generation:** Synthetic dataset consisting of 10,000 continuous readings mapped to 5 machines. Normal data is generated with Gaussian distributions, while anomalies (temperature spikes, pressure drops) are randomly injected into ~10% of records to simulate failure patterns.
2. **Preprocessing:** Train-test split (80/20), Standard Scaling of numerical features to normalize variance.
3. **Model Training:** Evaluation of Logistic Regression, Random Forest, and XGBoost based on F1-Score (chosen because classes are imbalanced; minimizing both False Positives and False Negatives is critical).
4. **Dashboarding:** Building a Streamlit application that loads the pre-trained model and scaler to infer real-time data inputs and plots visual reports using Plotly.

### 4. Tech Stack
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Frontend / Deployment:** Streamlit

### 5. Dataset Explanation
- `timestamp`: Time of the reading.
- `machine_id`: Identifier for the industrial machine (e.g., M_001).
- `temperature`: Operating temperature (°C).
- `vibration`: Machine vibration amplitude (mm/s).
- `pressure`: Fluid or air pressure (psi).
- `current` / `voltage`: Electrical characteristics.
- `rpm`: Rotations per minute.
- `humidity` / `load`: Environmental and operational stress factors.
- `failure_status`: Target variable (0 = Normal, 1 = Failure/Anomaly).

### 6. System Architecture
```mermaid
graph TD;
    A[IoT Sensors (Simulated Data)] --> B[Data Preprocessing & Scaling]
    B --> C[Machine Learning Model Selection]
    C --> D[Model Inference best_model.pkl]
    D --> E[Streamlit Dashboard Web App]
    E --> F{Output}
    F -->|Normal| G[Healthy Status]
    F -->|Anomaly Detected| H[Critical Warning & Maintenance Needed]
```

### 7. Resume Description
> **Predictive Maintenance System for Industrial IoT**
> * Python, Scikit-Learn, XGBoost, Streamlit
> - Engineered an end-to-end Machine Learning pipeline to predict equipment failure by synthesizing 10,000+ realistic IoT sensor datapoints (temperature, vibration, pressure).
> - Developed and evaluated multiple classification models (Random Forest, XGBoost), achieving high F1-Scores by correctly identifying anomalous patterns.
> - Deployed a real-time interactive Streamlit web application providing failure probabilities, actionable maintenance insights, and interactive data visualizations.

### 8. Viva Questions & Answers

**Q1: Why did you use synthetic data instead of real data?**
*Answer:* Gathering real industrial IoT data requires physical access to factory sensors, which is difficult due to confidentiality and safety reasons. Synthetic data allows us to model highly realistic physical environments and guarantees we have enough "failure" cases to adequately train the models without waiting for actual machines to break.

**Q2: Which machine learning model performed best and why?**
*Answer:* (Refer to the terminal output of your specific run). Generally, ensemble models like Random Forest or XGBoost perform best. They are capable of capturing non-linear relationships and interactions between multiple sensor features (like high temperature paired with low RPM), which simpler models like Logistic Regression often miss.

**Q3: Why did you scale the data using StandardScaler?**
*Answer:* Sensor values operate on vastly different scales (e.g., Vibration is ~1.5 mm/s, while RPM is ~1500). Machine learning models (especially gradient-descent based ones or models using distance metrics) can be skewed by features with larger magnitudes. Scaling ensures all features contribute proportionally to the model’s prediction.

**Q4: What metric did you use to evaluate the models? Why not just Accuracy?**
*Answer:* We prioritized the F1-Score (harmonic mean of Precision and Recall). In predictive maintenance, datasets are highly imbalanced (failures are rare, ~10%). A model that always predicts "Normal" would still get 90% accuracy but would be useless. F1-score ensures we correctly identify failures (Recall) without triggering too many false alarms (Precision).

**Q5: How does the Streamlit application predict real-time data?**
*Answer:* The Streamlit app takes the user's manual inputs (or conceptually, API streams), passes them through the saved `scaler.pkl` to normalize them exactly as the training data was normalized, and then infers the probability of failure using the saved `best_model.pkl`. It outputs a status and maintenance recommendation based on the predicted class probability.
