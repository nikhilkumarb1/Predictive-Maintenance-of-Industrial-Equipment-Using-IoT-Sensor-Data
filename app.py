import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide", page_icon="⚙️")

st.title("⚙️ Predictive Maintenance of Industrial Equipment")
st.markdown("### Real-Time IoT Sensor Data Analysis & Machine Health Prediction")

# Ensure models exist
try:
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except FileNotFoundError:
    st.error("Model or Scaler not found. Please run the model training script first.")
    st.stop()

# Sidebar for User Input
st.sidebar.header("Input Sensor Data")
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 150.0, 60.0)
vibration = st.sidebar.slider("Vibration (mm/s)", 0.0, 10.0, 1.5)
pressure = st.sidebar.slider("Pressure (psi)", 0.0, 200.0, 100.0)
current = st.sidebar.slider("Current (Amps)", 0.0, 50.0, 20.0)
voltage = st.sidebar.slider("Voltage (Volts)", 0.0, 300.0, 220.0)
rpm = st.sidebar.slider("RPM", 0.0, 3000.0, 1500.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 45.0)
load = st.sidebar.slider("Load (%)", 0.0, 100.0, 70.0)

# Input dataframe
input_data = pd.DataFrame({
    'temperature': [temperature],
    'vibration': [vibration],
    'pressure': [pressure],
    'current': [current],
    'voltage': [voltage],
    'rpm': [rpm],
    'humidity': [humidity],
    'load': [load]
})

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "Historical Data"])

with tab1:
    # Prediction logic
    st.subheader("Machine Health Prediction (Single)")
    
    if st.sidebar.button("Predict Machine Health"):
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 0:
                st.success("✅ Machine is Healthy.")
                st.metric(label="Failure Probability", value=f"{prediction_proba[1]*100:.2f}%")
            else:
                if prediction_proba[1] > 0.8:
                    st.error("🚨 Critical Alert: Machine Failure Imminent!")
                else:
                    st.warning("⚠️ Warning: Machine Anomaly Detected.")
                st.metric(label="Failure Probability", value=f"{prediction_proba[1]*100:.2f}%")
                
        with col2:
            st.markdown("### Maintenance Recommendation")
            if prediction == 0:
                st.info("- Continue normal operation.\n- Schedule routine maintenance as planned.")
            else:
                st.error("- Halt machine immediately to prevent damage.\n- Inspect cooling and vibration dampening systems.\n- Run a full diagnostic check.")

with tab2:
    st.subheader("Batch Machine Health Prediction")
    st.markdown("Upload a CSV or Excel dataset containing sensor readings to predict failures across multiple machines.")
    uploaded_file = st.file_uploader("Upload Sensor Dataset", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
            
            st.write("**Preview of Uploaded Data:**")
            st.dataframe(df_upload.head(3))
            
            required_cols = ['temperature', 'vibration', 'pressure', 'current', 'voltage', 'rpm', 'humidity', 'load']
            missing_cols = [col for col in required_cols if col not in df_upload.columns]
            
            if missing_cols:
                st.error(f"Uploaded dataset is missing the following required columns: {missing_cols}")
            else:
                if st.button("Run Batch Prediction"):
                    with st.spinner("Processing data and generating predictions..."):
                        pred_data = df_upload[required_cols]
                        pred_scaled = scaler.transform(pred_data)
                        predictions = model.predict(pred_scaled)
                        
                        # Add predictions to dataset
                        df_upload['Failure_Prediction'] = predictions
                        
                        failures_df = df_upload[df_upload['Failure_Prediction'] == 1]
                        
                        if 'machine_id' in df_upload.columns:
                            failed_machines = failures_df['machine_id'].unique()
                            if len(failed_machines) > 0:
                                st.error(f"🚨 Critical Alert: {len(failed_machines)} machine(s) predicted to experience failure!")
                                st.markdown("### Machines Requiring Maintenance")
                                st.info(", ".join([str(m) for m in failed_machines]))
                                
                                st.markdown("### Detailed Failure Records")
                                st.dataframe(failures_df)
                            else:
                                st.success("✅ All machines are predicted to be healthy! No maintenance required.")
                        else:
                            failures_count = len(failures_df)
                            if failures_count > 0:
                                st.error(f"🚨 Alert: {failures_count} instances of predicted failures detected!")
                                st.dataframe(failures_df)
                            else:
                                st.success("✅ All data records are predicted to be healthy!")
                                
        except Exception as e:
            st.error(f"Error reading or processing file: {e}")

with tab3:
    # Data Visualization Section
    st.subheader("Historical Data Visualization")
    data_path = "data/sensor_data.csv"
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        # Graphs
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Temperature vs Vibration (Normal vs Failure)**")
            fig1 = px.scatter(df.sample(min(1000, len(df)), random_state=42), x="temperature", y="vibration", color="failure_status",
                             color_continuous_scale=px.colors.diverging.RdYlGn[::-1],
                             labels={"failure_status": "Failure Status (1=Fail, 0=Normal)"})
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            st.markdown("**Feature Importance (From Trained Model)**")
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                features = input_data.columns
                indices = np.argsort(importances)[::-1]
                
                # Use pandas sort for easier plotly plotting
                feat_df = pd.DataFrame({'Feature': features[indices], 'Importance': importances[indices]}).sort_values('Importance', ascending=True)
                
                fig2 = px.bar(feat_df, x='Importance', y='Feature', orientation='h', color='Importance')
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Selected model does not support feature importance visualization directly.")
        
        st.markdown("---")
        st.markdown("**Sensor Correlation Heatmap**")
        
        # Select numeric columns for correlation (excluding ID columns)
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in ['machine_id', 'timestamp']]
        corr_matrix = df[numeric_cols].corr()
        
        fig3 = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', 
                         title="Correlation Between Different Sensors and Failure Status")
        st.plotly_chart(fig3, use_container_width=True)
        
    else:
        st.info("Run the data generation script to see historical data visualizations.")
