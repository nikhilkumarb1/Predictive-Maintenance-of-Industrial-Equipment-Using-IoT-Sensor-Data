import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Fallback to GradientBoosting if xgboost is missing
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False

def train_and_evaluate():
    # Load dataset
    data_path = "data/sensor_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Run data generator first.")
    
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    
    # Feature Selection
    features = ['temperature', 'vibration', 'pressure', 'current', 'voltage', 'rpm', 'humidity', 'load']
    X = df[features]
    y = df['failure_status']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        models["Gradient Boosting"] = GradientBoostingClassifier(random_state=42)
        
    print("\nTraining models...")
    
    best_model_name = ""
    best_f1 = 0
    best_model = None
    
    results = []
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model
            
    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df.to_string(index=False))
    
    print(f"\nBest Model based on F1-Score: {best_model_name} (F1: {best_f1:.4f})")
    print(classification_report(y_test, best_model.predict(X_test_scaled)))
    
    # Save best model and scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Model and Scaler saved successfully in 'models/' directory.")
    
    # Generate Feature Importance Plot for the best model
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        sns.barplot(x=importances[indices], y=[features[i] for i in indices], palette="viridis", hue=[features[i] for i in indices], legend=False)
        plt.xlabel("Relative Importance")
        os.makedirs("docs", exist_ok=True)
        plt.savefig("docs/feature_importance.png")
        plt.close()
        print("✅ Feature importance plot saved as 'docs/feature_importance.png'.")

if __name__ == "__main__":
    train_and_evaluate()
