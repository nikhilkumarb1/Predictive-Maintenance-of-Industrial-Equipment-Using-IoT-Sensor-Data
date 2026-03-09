import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# Load Model Files
# ===============================
model = joblib.load("predictive_maintenance_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("model_features.pkl")

st.set_page_config(page_title="Predictive Maintenance System", layout="wide")

st.title("Predictive Maintenance of Industrial Equipment")
st.write("IoT Sensor Based Machine Failure Prediction Dashboard")

# ===============================
# Upload Multiple Files
# ===============================
uploaded_files = st.file_uploader(
    "Upload Machine Sensor Data (Excel)",
    type=["xlsx"],
    accept_multiple_files=True
)

all_results = []

if uploaded_files:

    for file in uploaded_files:

        st.subheader(f"Processing File: {file.name}")

        df = pd.read_excel(file)
        original_df = df.copy()

        st.write("Preview")
        st.dataframe(df.head())

        # ===============================
        # Preprocessing
        # ===============================
        if "Defect" in df.columns:
            df["Defect"] = df["Defect"].fillna(0)

        if "TIMESTAMP" in df.columns:
            df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")

        drop_cols = ["Date", "TIMESTAMP"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        df = df.fillna(df.median(numeric_only=True))
        df = pd.get_dummies(df, drop_first=True)

        # Match training features
        df = df.reindex(columns=feature_columns, fill_value=0)

        # Scale
        scaled = scaler.transform(df)

        # Predict
        pred = model.predict(scaled)
        pred = np.where(pred == -1, 1, 0)

        original_df["Failure_Prediction"] = pred
        original_df["Machine Status"] = original_df["Failure_Prediction"].map({
            0: "Healthy",
            1: "Failure Risk"
        })

        all_results.append(original_df)

        st.write("Prediction Results")
        st.dataframe(original_df)

    # ===============================
    # Combine All Machines
    # ===============================
    final_df = pd.concat(all_results, ignore_index=True)

    st.header("Overall Factory Health Summary")

    healthy = (final_df["Machine Status"] == "Healthy").sum()
    failure = (final_df["Machine Status"] == "Failure Risk").sum()

    col1, col2 = st.columns(2)

    col1.metric("Healthy Machines", healthy)
    col2.metric("Failure Risk Machines", failure)

    # ===============================
    # Chart
    # ===============================
    chart_data = pd.DataFrame({
        "Status": ["Healthy", "Failure Risk"],
        "Count": [healthy, failure]
    })

    st.subheader("Machine Health Distribution")
    st.bar_chart(chart_data.set_index("Status"))

    # ===============================
    # Failure Detection Table
    # ===============================
    st.subheader("Machines at Risk")
    risk_df = final_df[final_df["Machine Status"] == "Failure Risk"]

    if len(risk_df) > 0:
        st.dataframe(risk_df)
    else:
        st.success("No Machine Failure Risk Detected")

    # ===============================
    # Download Results
    # ===============================
    csv = final_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download All Prediction Results",
        data=csv,
        file_name="machine_health_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Upload one or more machine datasets to start monitoring.")