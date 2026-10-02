import pandas as pd
import numpy as np
import time
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)

# 1. Strict ECE Implementation (From Step 1 Audit)
def compute_ece_strict(y_true, y_proba, n_bins=15):
    confidences = np.max(y_proba, axis=1)
    predictions = np.argmax(y_proba, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0.0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

# 2. True Single-Sample Latency
def measure_single_sample_latency(model, X_val, n_samples=200):
    # Warmup
    for i in range(10):
        model.predict_proba(X_val.iloc[[i]])
    
    latencies = []
    # Test on limited samples to keep MOO fast
    for i in range(min(n_samples, len(X_val))):
        sample = X_val.iloc[[i]]
        t0 = time.time()
        model.predict_proba(sample)
        t1 = time.time()
        latencies.append((t1 - t0) * 1000) # ms
    return np.median(latencies)

def run_step3():
    print("--- Step 3: Improved MOO Pipeline (NSGA-II) ---")
    
    # 1. Load and Strict Deduplication
    df = pd.read_csv("/Users/devrajsinghal/Desktop/capstone/Data/CICIDS2017_sample_km.csv")
    df = df.drop_duplicates().reset_index(drop=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # 2. Strict Train (60%), Val (20%), Test (20%) Split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
    
    # Assert no leakage
    assert len(set(X_train.index).intersection(set(X_val.index))) == 0
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0
    assert len(set(X_val.index).intersection(set(X_test.index))) == 0
    
    print(f"Shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 3. Feature Selection Ablation (Information Gain vs SHAP/Embedded)
    print("\n--- Running Feature Selection ---")
    # IG
    ig = mutual_info_classif(X_train.sample(5000, random_state=42), y_train.sample(5000, random_state=42))
    ig_top_20 = X_train.columns[np.argsort(ig)[-20:]].tolist()
    
    # Embedded (LightGBM split counts/gains)
    clf_fs = LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)
    clf_fs.fit(X_train, y_train)
    embedded_top_20 = X_train.columns[np.argsort(clf_fs.feature_importances_)[-20:]].tolist()
    
    print(f"IG Top 20 overlap with Embedded Top 20: {len(set(ig_top_20).intersection(set(embedded_top_20)))} features")
    
    # We will use ALL features for the full MOO to give Optuna the maximum ceiling, 
    # but acknowledge FS in the paper.
    
    # 4. Optuna MOO with NSGA-II
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'class_weight': 'balanced',
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        clf = LGBMClassifier(**params)
        clf.fit(X_train, y_train)
        
        preds = clf.predict(X_val)
        proba = clf.predict_proba(X_val)
        
        macro_f1 = f1_score(y_val, preds, average='macro')
        error = 1.0 - macro_f1 # Minimize error
        
        ece = compute_ece_strict(y_val, proba)
        latency = measure_single_sample_latency(clf, X_val, n_samples=100)
        
        return error, ece, latency

    print("\n--- Starting NSGA-II MOO (50 trials) ---")
    sampler = optuna.samplers.NSGAIISampler(seed=42)
    study = optuna.create_study(directions=["minimize", "minimize", "minimize"], sampler=sampler)
    study.optimize(objective, n_trials=50, timeout=300)
    
    # 5. Extract Pareto Front
    pareto_trials = study.best_trials
    print(f"\nNumber of models on the Pareto Front: {len(pareto_trials)}")
    
    errors = [t.values[0] for t in pareto_trials]
    eces = [t.values[1] for t in pareto_trials]
    lats = [t.values[2] for t in pareto_trials]
    
    # 6. Model Selection Rule:
    # Rule: Pick the model with Latency < 1.0 ms and minimum Error. If none, pick min Error.
    valid_trials = [t for t in pareto_trials if t.values[2] < 1.0]
    if len(valid_trials) > 0:
        best_trial = min(valid_trials, key=lambda t: t.values[0])
        print("Selected model based on latency budget < 1.0 ms.")
    else:
        best_trial = min(pareto_trials, key=lambda t: t.values[0])
        print("Selected model based on minimum Error (latency budget failed).")
        
    print(f"Selected Validation Specs -> Error: {best_trial.values[0]:.4f} (Macro-F1: {1-best_trial.values[0]:.4f}), ECE: {best_trial.values[1]:.4f}, Latency: {best_trial.values[2]:.4f} ms")
    print(f"Hyperparameters: {best_trial.params}")
    
    # 7. Final Test Evaluation
    print("\n--- Final Test Set Evaluation ---")
    final_params = best_trial.params
    final_params['class_weight'] = 'balanced'
    final_params['random_state'] = 42
    final_params['verbose'] = -1
    final_params['n_jobs'] = -1
    
    final_clf = LGBMClassifier(**final_params)
    final_clf.fit(X_train, y_train)
    
    test_preds = final_clf.predict(X_test)
    test_proba = final_clf.predict_proba(X_test)
    
    test_macro_f1 = f1_score(y_test, test_preds, average='macro')
    test_ece = compute_ece_strict(y_test, test_proba)
    test_latency = measure_single_sample_latency(final_clf, X_test, n_samples=1000)
    
    print(f"Final Test Macro-F1: {test_macro_f1:.4f}")
    print(f"Final Test ECE: {test_ece:.4f}")
    print(f"Final Test Single-Sample Latency (Median): {test_latency:.4f} ms")
    
    # Write summary to file
    with open("/Users/devrajsinghal/Desktop/capstone/improved/results/step3_improved.md", "w") as f:
        f.write("# Step 3: Improved MOO Pipeline (NSGA-II)\n\n")
        f.write("## Pareto Front Model Selection\n")
        f.write(f"Optuna found {len(pareto_trials)} models on the Pareto Front across (Error, ECE, Latency).\n")
        f.write("We selected the model strictly based on the Validation split with the rule: Minimize Error where Latency < 1.0 ms.\n\n")
        f.write("### Chosen Hyperparameters:\n")
        f.write(f"```json\n{best_trial.params}\n```\n\n")
        f.write("## Final Hold-out Test Set Performance\n")
        f.write(f"- **Macro-F1**: {test_macro_f1:.4f}\n")
        f.write(f"- **ECE**: {test_ece:.4f}\n")
        f.write(f"- **Single-Sample Latency (Median)**: {test_latency:.4f} ms\n")

if __name__ == "__main__":
    run_step3()
