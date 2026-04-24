import pandas as pd
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Data
print("Loading dataset...")
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# 2. SMOTE: Create synthetic fraud cases to balance the 0.17%
print("Applying SMOTE to balance the data...")
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# 4. XGBoost Classifier
print("Training Advanced XGBoost Model...")
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# 5. Results
y_pred = model.predict(X_test)
print("\n--- ADVANCED RESULTS (SMOTE + XGBOOST) ---")
print(classification_report(y_test, y_pred))