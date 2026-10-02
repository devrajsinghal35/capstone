import os
import time
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.calibration import calibration_curve

# Directory setup
os.makedirs("results", exist_ok=True)

# Helper function for ECE
def compute_ece(y_true, y_proba, n_bins=10):
    y_pred_class = np.argmax(y_proba, axis=1)
    correct = (np.array(y_true) == y_pred_class).astype(int)
    confidence = np.max(y_proba, axis=1)
    prob_true, prob_pred = calibration_curve(correct, confidence, n_bins=n_bins)
    bin_counts, _ = np.histogram(confidence, bins=n_bins, range=(0.0, 1.0))
    bin_weights = bin_counts[:len(prob_true)] / np.sum(bin_counts)
    ece = np.sum(np.abs(prob_true - prob_pred) * bin_weights)
    return ece

# Load data and remove duplicates to fix leakage
df = pd.read_csv("/Users/devrajsinghal/Desktop/capstone/Data/CICIDS2017_sample_km.csv")
print("Original shape:", df.shape)
df = df.drop_duplicates().reset_index(drop=True)
print("Deduplicated shape:", df.shape)

X = df.drop('Label', axis=1)
y = df['Label']

# Train/Test Split (80/20)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Corrected Baseline: Default LightGBM on deduplicated data
print("\n--- Running Corrected Baseline (LightGBM) ---")
baseline_f1s = []
baseline_accs = []
baseline_eces = []
baseline_lats = []

for seed in [42, 43, 44, 45, 46]:
    clf_base = lgb.LGBMClassifier(random_state=seed, verbose=-1, n_jobs=-1)
    clf_base.fit(X_train_full, y_train_full)
    
    t1 = time.time()
    preds_base = clf_base.predict(X_test)
    t2 = time.time()
    
    proba_base = clf_base.predict_proba(X_test)
    
    f1 = f1_score(y_test, preds_base, average='weighted')
    acc = accuracy_score(y_test, preds_base)
    ece = compute_ece(y_test, proba_base)
    lat = ((t2 - t1) / len(X_test)) * 1000 # ms per sample
    
    baseline_f1s.append(f1)
    baseline_accs.append(acc)
    baseline_eces.append(ece)
    baseline_lats.append(lat)

print(f"Corrected Baseline F1: {np.mean(baseline_f1s):.4f} ± {np.std(baseline_f1s):.4f}")
print(f"Corrected Baseline Accuracy: {np.mean(baseline_accs):.4f} ± {np.std(baseline_accs):.4f}")
print(f"Corrected Baseline ECE: {np.mean(baseline_eces):.4f} ± {np.std(baseline_eces):.4f}")
print(f"Corrected Baseline Latency: {np.mean(baseline_lats):.4f} ± {np.std(baseline_lats):.4f} ms")


# Improved Method: True MOO (NSGA-II via Optuna) on Validation Set
print("\n--- Running Improved Method (True MOO via Optuna NSGA-II) ---")
# Use a validation set for tuning to avoid test set leakage
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
)

def objective(trial):
    # Hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 50),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']), # Replace oversampling with class weights
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_train, y_train)
    
    # Inference on Val
    t1 = time.time()
    preds = clf.predict(X_val)
    t2 = time.time()
    
    proba = clf.predict_proba(X_val)
    
    f1 = f1_score(y_val, preds, average='weighted')
    ece = compute_ece(y_val, proba)
    latency = ((t2 - t1) / len(X_val)) * 1000 # ms per sample
    
    return f1, ece, latency

study = optuna.create_study(directions=['maximize', 'minimize', 'minimize'])
study.optimize(objective, n_trials=50, timeout=600)

print(f"Number of trials on the Pareto front: {len(study.best_trials)}")
best_f1_trial = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)[0]
best_params = best_f1_trial.params
print("Best Params (Max F1 from Pareto):", best_params)

# Multi-seed evaluation of best improved model
improved_f1s = []
improved_accs = []
improved_eces = []
improved_lats = []

for seed in [42, 43, 44, 45, 46]:
    params = best_params.copy()
    params['random_state'] = seed
    params['verbose'] = -1
    params['n_jobs'] = -1
    
    clf_opt = lgb.LGBMClassifier(**params)
    clf_opt.fit(X_train_full, y_train_full)
    
    t1 = time.time()
    preds_opt = clf_opt.predict(X_test)
    t2 = time.time()
    
    proba_opt = clf_opt.predict_proba(X_test)
    
    f1 = f1_score(y_test, preds_opt, average='weighted')
    acc = accuracy_score(y_test, preds_opt)
    ece = compute_ece(y_test, proba_opt)
    lat = ((t2 - t1) / len(X_test)) * 1000
    
    improved_f1s.append(f1)
    improved_accs.append(acc)
    improved_eces.append(ece)
    improved_lats.append(lat)

print(f"Improved Method F1: {np.mean(improved_f1s):.4f} ± {np.std(improved_f1s):.4f}")
print(f"Improved Method Accuracy: {np.mean(improved_accs):.4f} ± {np.std(improved_accs):.4f}")
print(f"Improved Method ECE: {np.mean(improved_eces):.4f} ± {np.std(improved_eces):.4f}")
print(f"Improved Method Latency: {np.mean(improved_lats):.4f} ± {np.std(improved_lats):.4f} ms")

# Save results
res_df = pd.DataFrame({
    'Model': ['Corrected Baseline']*5 + ['Improved True MOO']*5,
    'Seed': [42, 43, 44, 45, 46]*2,
    'Accuracy': baseline_accs + improved_accs,
    'F1': baseline_f1s + improved_f1s,
    'ECE': baseline_eces + improved_eces,
    'Latency_ms': baseline_lats + improved_lats
})
res_df.to_csv("results/comparison_results.csv", index=False)

# Pareto Front Plot
pareto_f1 = [t.values[0] for t in study.best_trials]
pareto_ece = [t.values[1] for t in study.best_trials]
pareto_lat = [t.values[2] for t in study.best_trials]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pareto_f1, pareto_ece, pareto_lat, c='r', marker='o')
ax.set_xlabel('F1 Score (Maximize)')
ax.set_ylabel('ECE (Minimize)')
ax.set_zlabel('Latency ms (Minimize)')
plt.title('Pareto Front (F1 vs ECE vs Latency)')
plt.savefig("results/pareto_front_3d.png")

# Confusion Matrix for Best Model (seed 42)
cm = confusion_matrix(y_test, preds_opt)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Improved Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("results/improved_cm.png")

print("All experiments completed. Results saved to results/ directory.")
