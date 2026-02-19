import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE
import joblib
import os

def train_model():
    print("Loading data...")
    try:
        df = pd.read_csv("data/lending_data.csv")
    except FileNotFoundError:
        print("Error: data/lending_data.csv not found. Run src/generate_data.py first.")
        return

    # Features and Target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    # SMOTE for Imbalanced Data
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Model Training with XGBoost
    print("Training XGBoost Classifier...")
    # scale_pos_weight is useful for imbalanced datasets, though SMOTE helps too.
    # XGBoost is state-of-the-art for tabular data.
    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        use_label_encoder=False, 
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    
    print(f"Model Evaluation (XGBoost + SMOTE):")
    print(f"Accuracy: {acc:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"Precision: {prec:.2f}")
    
    # Save Model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/credit_risk_model.pkl")
    print("Model saved to model/credit_risk_model.pkl")

if __name__ == "__main__":
    train_model()
