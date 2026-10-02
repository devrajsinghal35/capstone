import pandas as pd
import numpy as np
import time
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, balanced_accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

def run_step2():
    df = pd.read_csv("/Users/devrajsinghal/Desktop/capstone/Data/CICIDS2017_sample_km.csv")
    df_dedup = df.drop_duplicates().reset_index(drop=True)
    df_dedup.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_dedup.fillna(0, inplace=True)
    
    X = df_dedup.drop('Label', axis=1)
    y = df_dedup['Label']
    
    methods = [
        "Default", 
        "Class_Weight_Balanced", 
        "SMOTE_Only", 
        "Buggy_SMOTE_ADASYN", 
        "Fixed_SMOTE_ADASYN"
    ]
    models = ["LightGBM", "XGBoost"]
    seeds = range(42, 52)
    
    results = []
    
    for seed in seeds:
        print(f"--- Running Seed {seed} ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )
        assert len(set(X_train.index).intersection(set(X_test.index))) == 0
        
        # Precompute balanced datasets to save time
        # 3. SMOTE Only
        smote = SMOTE(random_state=seed)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # 4. Buggy SMOTE+ADASYN
        counter_y = Counter(y_train)
        avg_samples = sum(counter_y.values()) / len(counter_y)
        target = int(avg_samples / 2)
        minority_classes = {k: target for k, v in counter_y.items() if v < target}
        
        X_train_buggy, y_train_buggy = X_train.copy(), y_train.copy()
        if minority_classes:
            min_samples = min(counter_y.values())
            k_neighbors = min(5, max(1, min_samples - 1))
            smote_buggy = SMOTE(sampling_strategy=minority_classes, k_neighbors=k_neighbors, random_state=seed)
            X_train_buggy, y_train_buggy = smote_buggy.fit_resample(X_train_buggy, y_train_buggy)
            try:
                adasyn_buggy = ADASYN(sampling_strategy=minority_classes, n_neighbors=k_neighbors, random_state=seed)
                X_train_buggy, y_train_buggy = adasyn_buggy.fit_resample(X_train_buggy, y_train_buggy)
            except Exception:
                pass # ADASYN does nothing because targets are already met
                
        # 5. Fixed SMOTE+ADASYN
        X_train_fixed, y_train_fixed = X_train.copy(), y_train.copy()
        if minority_classes:
            # Generate 50% of the required difference via SMOTE, then rest via ADASYN
            # Since ADASYN dynamically generates, we just ask it to reach the final target
            smote_target = {k: v + int((target - v) * 0.5) for k, v in counter_y.items() if k in minority_classes}
            smote_fixed = SMOTE(sampling_strategy=smote_target, k_neighbors=k_neighbors, random_state=seed)
            X_train_fixed, y_train_fixed = smote_fixed.fit_resample(X_train_fixed, y_train_fixed)
            
            try:
                adasyn_fixed = ADASYN(sampling_strategy=minority_classes, n_neighbors=k_neighbors, random_state=seed)
                X_train_fixed, y_train_fixed = adasyn_fixed.fit_resample(X_train_fixed, y_train_fixed)
            except Exception:
                pass
        
        train_sets = {
            "Default": (X_train, y_train),
            "Class_Weight_Balanced": (X_train, y_train),
            "SMOTE_Only": (X_train_smote, y_train_smote),
            "Buggy_SMOTE_ADASYN": (X_train_buggy, y_train_buggy),
            "Fixed_SMOTE_ADASYN": (X_train_fixed, y_train_fixed)
        }
        
        for model_name in models:
            for method in methods:
                X_tr, y_tr = train_sets[method]
                
                if model_name == "LightGBM":
                    if method == "Class_Weight_Balanced":
                        clf = LGBMClassifier(class_weight='balanced', random_state=seed, verbose=-1, n_jobs=-1)
                    else:
                        clf = LGBMClassifier(random_state=seed, verbose=-1, n_jobs=-1)
                else:
                    if method == "Class_Weight_Balanced":
                        # XGBoost heuristic for multiclass weights: compute sample weights
                        from sklearn.utils.class_weight import compute_sample_weight
                        sample_weights = compute_sample_weight('balanced', y_tr)
                        clf = XGBClassifier(random_state=seed, objective="multi:softprob", eval_metric="mlogloss", n_jobs=-1)
                    else:
                        clf = XGBClassifier(random_state=seed, objective="multi:softprob", eval_metric="mlogloss", n_jobs=-1)

                t1 = time.time()
                if method == "Class_Weight_Balanced" and model_name == "XGBoost":
                    clf.fit(X_tr, y_tr, sample_weight=sample_weights)
                else:
                    clf.fit(X_tr, y_tr)
                t2 = time.time()
                
                preds = clf.predict(X_test)
                
                acc = accuracy_score(y_test, preds)
                f1_mac = f1_score(y_test, preds, average='macro')
                f1_wt = f1_score(y_test, preds, average='weighted')
                mcc = matthews_corrcoef(y_test, preds)
                bal_acc = balanced_accuracy_score(y_test, preds)
                
                results.append({
                    "Seed": seed,
                    "Model": model_name,
                    "Method": method,
                    "Accuracy": acc,
                    "Macro_F1": f1_mac,
                    "Weighted_F1": f1_wt,
                    "MCC": mcc,
                    "Bal_Acc": bal_acc,
                    "Train_Time": t2 - t1
                })

    res_df = pd.DataFrame(results)
    
    # Calculate Mean ± Std
    summary = res_df.groupby(['Model', 'Method']).agg(
        Macro_F1=('Macro_F1', lambda x: f"{x.mean():.4f} ± {x.std():.4f}"),
        Weighted_F1=('Weighted_F1', lambda x: f"{x.mean():.4f} ± {x.std():.4f}"),
        Accuracy=('Accuracy', lambda x: f"{x.mean():.4f} ± {x.std():.4f}"),
        MCC=('MCC', lambda x: f"{x.mean():.4f} ± {x.std():.4f}"),
        Bal_Acc=('Bal_Acc', lambda x: f"{x.mean():.4f} ± {x.std():.4f}"),
        Train_Time=('Train_Time', lambda x: f"{x.mean():.2f}s")
    ).reset_index()
    
    # Save to markdown
    with open("/Users/devrajsinghal/Desktop/capstone/improved/results/step2_fair_comparison.md", "w") as f:
        f.write("# Step 2: Fair Comparison (10 Seeds, Leak-Free)\n\n")
        f.write("Does oversampling help compared to native class weighting?\n\n")
        f.write(summary.to_markdown(index=False))
        
        f.write("\n\n## Verification of Paper's Claims\n")
        f.write("- **Test Set Used During Tuning**: `fast_reproduce.py` lines 198-199 explicitly call `clf.fit(X_tr, y_tr)` and `preds = clf.predict(X_te)` directly inside `lgb_mopso`, where `X_te` is the global `X_test`.\n")
        f.write("- **Linear Scalarization instead of Pareto**: `fast_reproduce.py` line 205 calculates fitness as `np.array([-0.90 * f1, -0.05 * conf, 0.05 * elapsed])` and lines 207-208 just sum it: `if np.sum(fitness) < np.sum(...)`.\n")
        f.write("- **Latency measured as Training Time**: `fast_reproduce.py` lines 196-203 measures `start_t = time.time()`, trains the model `clf.fit(X_tr, y_tr)`, tests it, and then sets `elapsed = time.time() - start_t`. This mixes training time and batch testing time, not single-sample latency.\n")

if __name__ == "__main__":
    run_step2()
