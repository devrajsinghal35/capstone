import pandas as pd
import numpy as np
import time
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, balanced_accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def run_step1():
    print("--- STEP 1.1: Baseline Audit ---")
    
    # 1. Load Data
    df = pd.read_csv("/Users/devrajsinghal/Desktop/capstone/Data/CICIDS2017_sample_km.csv")
    print(f"Original shape: {df.shape}")
    print("Original Class Counts:")
    print(df['Label'].value_counts())
    
    # NaN/Inf handling
    print("\nCheck NaNs and Infs before cleanup:")
    num_nans = df.isna().sum().sum()
    num_infs = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    print(f"NaNs: {num_nans}, Infs: {num_infs}")
    
    if num_nans > 0 or num_infs > 0:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
    
    # 2. Deduplication
    df_dedup = df.drop_duplicates().reset_index(drop=True)
    print(f"\nDeduplicated shape: {df_dedup.shape}")
    print("Deduplicated Class Counts:")
    print(df_dedup['Label'].value_counts())
    
    X = df_dedup.drop('Label', axis=1)
    y = df_dedup['Label']
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print("\nTrain Class Counts:")
    print(y_train.value_counts())
    print("\nTest Class Counts:")
    print(y_test.value_counts())
    
    # 4. Compare XGBoost, RF, LightGBM
    print("\n--- Model Comparison on Deduplicated Split ---")
    
    models = {
        "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(random_state=42, objective="multi:softprob", eval_metric="mlogloss", n_jobs=-1),
        "LightGBM (default)": LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
        "LightGBM (min_child_samples=5)": LGBMClassifier(random_state=42, min_child_samples=5, verbose=-1, n_jobs=-1)
    }
    
    for name, model in models.items():
        t1 = time.time()
        model.fit(X_train, y_train)
        t2 = time.time()
        preds = model.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average='macro')
        f1_weighted = f1_score(y_test, preds, average='weighted')
        
        print(f"[{name}] Acc: {acc:.4f} | F1-Macro: {f1_macro:.4f} | F1-Weighted: {f1_weighted:.4f} | Train Time: {t2-t1:.2f}s")
        if name == "LightGBM (default)":
            print("LightGBM Classification Report:")
            print(classification_report(y_test, preds))

if __name__ == "__main__":
    run_step1()
