# Fintech Sentinel: Dual-Engine Fraud Detection System
**Srijan S. Kotian | AI Research Portfolio**

## 📊 Project Overview
Fintech Sentinel is a hybrid AI monitoring system designed to identify fraudulent transactions in highly imbalanced financial datasets. Unlike standard classifiers, this system utilizes a **Dual-Engine Architecture** to detect both known fraud patterns and novel anomalies.

## 🔬 Research Methodology
- **Class Imbalance Handling:** Utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to address the 0.17% fraud prevalence, ensuring the model learns minority class features.
- **Hybrid Architecture:** 1. **Supervised:** Random Forest Classifier for historical signature matching.
    2. **Unsupervised:** Isolation Forest for zero-day anomaly detection.
- **Feature Engineering:** Implemented **Robust Scaling** to mitigate the impact of transaction amount outliers.

## 📈 Performance Benchmarks
- **AUPRC (Area Under Precision-Recall Curve):** 0.9939
- **Inference Latency:** ~109ms (Optimized for real-time banking throughput).
- **Validation:** Evaluated on 284k+ transactions with a focus on minimizing False Negatives.

## 🚀 Deployment
The system is live and interactive.
- **Live Demo:** [Insert your Streamlit Link Here]
- **Stack:** Python, Scikit-Learn, Streamlit, Joblib.

## 🛠️ Installation & Usage
1. Clone the repo: `git clone https://github.com/srijan-kotian/AI_Research_Portfolio.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the monitor: `streamlit run app.py`