import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# Load the "Supervised" and "Unsupervised" brains
model = joblib.load('fraud_model.pkl')
anomaly_detector = joblib.load('anomaly_detector.pkl')

st.set_page_config(page_title="Fintech Sentinel AI", page_icon="💳", layout="wide")

st.title("💳 Fintech Sentinel: Advanced Fraud Detection")
st.markdown("---")

# Sidebar for Transaction Input
st.sidebar.header("📥 Transaction Entry")
amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=120.50)
time_step = st.sidebar.slider("Seconds since last event", 0, 172800, 3600)

# We need 30 features total (V1-V28 + Time + Amount)
# For the demo, we generate neutral PCA features for V1-V28
pca_features = [0.0] * 28 

if st.sidebar.button("Analyze Transaction"):
    # Combine inputs into the 30-feature vector
    input_data = np.array([[time_step] + pca_features + [amount]])
    
    with st.spinner('Running Dual-Model Verification...'):
        start = time.time()
        
        # 1. Supervised Prediction
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        
        # 2. Unsupervised Anomaly Check
        is_anomaly = anomaly_detector.predict(input_data)[0]
        
        latency = (time.time() - start) * 1000

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Fraud Probability", f"{prob*100:.2f}%")
        
    with col2:
        status = "⚠️ HIGH RISK" if prediction == 1 else "✅ SECURE"
        st.subheader(f"Status: {status}")
        
    with col3:
        st.metric("Latency", f"{latency:.2f} ms")

    st.markdown("---")
    
    # Research Insight
    if is_anomaly == -1:
        st.warning("🕵️ **Unsupervised Alert:** Isolation Forest flagged this as an Outlier (Anomaly).")
    else:
        st.success("✔️ **Behavioral Check:** Transaction patterns match historical normal clusters.")

st.info("**Researcher Note:** This system employs a Dual-Model architecture (Random Forest + Isolation Forest) to maximize recall while maintaining low inference latency.")