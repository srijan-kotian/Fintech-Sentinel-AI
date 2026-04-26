import pandas as pd
import numpy as np
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# --- 1. DATA INGESTION & PRE-PROCESSING ---
print("🚀 Loading Financial Dataset...")
df = pd.read_csv('data/creditcard.csv')

# RobustScaler is essential for financial data to handle extreme transaction outliers
scaler = RobustScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

X = df.drop('Class', axis=1)
y = df['Class']

# --- 2. HANDLING EXTREME IMBALANCE (SMOTE) ---
# Synthetic Minority Over-sampling to balance the 0.17% fraud cases
print("⚖️ Applying SMOTE for Class Balancing...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# --- 3. SUPERVISED LEARNING: RANDOM FOREST ---
print("🌲 Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)

# --- 4. UNSUPERVISED LEARNING: ISOLATION FOREST ---
# This detects 'outliers' without needing labels—highly valued by researchers
print("🕵️ Running Unsupervised Anomaly Detection (Isolation Forest)...")
iso_forest = IsolationForest(n_estimators=100, contamination=0.0017, random_state=42)
iso_forest.fit(X) # Unsupervised: No 'y' used here

# --- 5. SYSTEM PERFORMANCE & LATENCY ANALYSIS ---
print("\n" + "="*30)
print("📊 RESEARCH PERFORMANCE REPORT")
print("="*30)

# Evaluate Supervised Model
y_pred = rf_model.predict(X_test)
auprc = average_precision_score(y_test, y_pred)
print(f"Area Under Precision-Recall Curve (AUPRC): {auprc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Measure Latency (Critical for Fintech Engineering)
start_time = time.time()
sample_pred = rf_model.predict(X_test[:1])
latency = (time.time() - start_time) * 1000
print(f"⏱️ Inference Latency: {latency:.2f} ms")

# --- 6. SERIALIZATION FOR DEPLOYMENT ---
print("\n💾 Saving models for Streamlit deployment...")
joblib.dump(rf_model, 'fraud_model.pkl')
joblib.dump(iso_forest, 'anomaly_detector.pkl')
print("✅ Done! Files 'fraud_model.pkl' and 'anomaly_detector.pkl' created.")