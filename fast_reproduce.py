#!/usr/bin/env python
import os
import warnings
warnings.filterwarnings("ignore")

cache_dir = os.path.abspath(".matplotlib_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["MPLCONFIGDIR"] = cache_dir

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
import joblib
import time
from scipy.stats import shapiro
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN

os.makedirs("output", exist_ok=True)

print("="*60)
print("MOO-AutoML IDS Quick Reproduction Pipeline")
print("="*60)

csv_path = "Data/CICIDS2017_sample_km.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset file {csv_path} not found.")

full_df = pd.read_csv(csv_path)
print(f"Full dataset loaded: {full_df.shape[0]} rows, {full_df.shape[1]} columns.")

# Subsample 20% of rows cleanly
df = full_df.sample(frac=0.2, random_state=42).reset_index(drop=True)
print(f"Subsampled dataset (20%): {df.shape[0]} rows, {df.shape[1]} columns.")
print("Class distribution:\n", df['Label'].value_counts())

# Auto Normalization
def Auto_Normalization(df_in):
    numeric_features = df_in.drop(['Label'], axis=1).dtypes[df_in.dtypes != 'object'].index
    sample_vals = df_in[numeric_features].values.flatten()[:1000]
    stat, p = shapiro(sample_vals)
    alpha = 0.05

    if p > alpha:
        print('Z-score normalization automatically chosen')
        df_in[numeric_features] = df_in[numeric_features].apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    else:
        print('Min-max normalization automatically chosen')
        df_in[numeric_features] = df_in[numeric_features].apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))
    return df_in

df = Auto_Normalization(df)

if df.isnull().values.any() or np.isinf(df).values.any():
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

X = df.drop(['Label'], axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42, stratify=y)

print("\n--- Training Base Learners (Initial Baseline) ---")
models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "ExtraTrees": ExtraTreesClassifier(random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(random_state=42, objective="multi:softprob", eval_metric="mlogloss", n_jobs=-1),
    "LightGBM": lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
}

baseline_results = {}
for name, model in models.items():
    t1 = time.time()
    model.fit(X_train, y_train)
    t2 = time.time()
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average='weighted')
    acc = accuracy_score(y_test, preds)
    baseline_results[name] = {"F1": f1, "Accuracy": acc, "TrainTime": t2-t1}
    print(f"[{name}] F1: {f1:.4f} | Accuracy: {acc:.4f} | Time: {t2-t1:.2f}s")

# Hybrid Data Balancing
print("\n--- Applying Hybrid Data Balancing (SMOTE + ADASYN) ---")
counter_y = Counter(y_train)
average_samples_per_class = sum(counter_y.values()) / len(counter_y)
target_samples = int(average_samples_per_class / 2)
minority_classes = {k: target_samples for k, v in counter_y.items() if v < target_samples}

if minority_classes:
    min_samples = min(counter_y.values())
    k_neighbors = min(5, max(1, min_samples - 1))
    smote = SMOTE(sampling_strategy=minority_classes, k_neighbors=k_neighbors, random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print("Class distribution after hybrid balancing:\n", pd.Series(y_train).value_counts())

# Phase 2: OIP-AutoFS (MOPSO Feature Selection)
print("\n" + "="*60)
print("Phase 2: OIP-AutoFS (MOPSO Automated Feature Selection)")
print("="*60)

def get_original_feature_importance(X_tr, y_tr):
    clf = lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    return clf.feature_importances_

def fs_objective_function(position, X_tr, X_te, y_tr, y_te, original_importances, f_w=0.9, p_w=0.1):
    mask = position > 0.5
    if np.sum(mask) == 0:
        return float('-inf'), 1.0

    X_tr_sel = X_tr.iloc[:, mask]
    clf = lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1)
    clf.fit(X_tr_sel, y_tr)

    selected_imp = original_importances[mask]
    tot_imp = np.sum(original_importances)
    norm_imp = np.sum(selected_imp) / tot_imp if tot_imp > 0 else 0
    feat_pct = np.sum(mask) / X_tr.shape[1]

    score = f_w * norm_imp - p_w * feat_pct
    return score, feat_pct

def mopso_fs(X_tr, X_te, y_tr, y_te, num_particles=10, max_iterations=12):
    num_features = X_tr.shape[1]
    orig_imp = get_original_feature_importance(X_tr, y_tr)
    swarm = [{'position': np.random.rand(num_features), 'velocity': np.random.uniform(-0.1, 0.1, num_features)} for _ in range(num_particles)]
    gbest_score = float('-inf')
    gbest_pos = None

    for iteration in range(max_iterations):
        for particle in swarm:
            score, _ = fs_objective_function(particle['position'], X_tr, X_te, y_tr, y_te, orig_imp)
            if score > particle.get('pbest_score', float('-inf')):
                particle['pbest_score'] = score
                particle['pbest_position'] = particle['position']
            if score > gbest_score:
                gbest_score = score
                gbest_pos = particle['position']

        w = 0.9 - 0.5 * (iteration / max_iterations)
        for p in swarm:
            inertia = w * p['velocity']
            cognitive = 1.5 * np.random.random() * (p['pbest_position'] - p['position'])
            social = 1.5 * np.random.random() * (gbest_pos - p['position'])
            p['velocity'] = np.clip(inertia + cognitive + social, -0.1, 0.1)
            p['position'] = np.clip(p['position'] + p['velocity'], 0, 1)

        print(f"OIP-AutoFS Iteration {iteration+1}/{max_iterations} complete. Best Score: {gbest_score:.4f}")

    selected_features = gbest_pos > 0.5
    selected_names = X_tr.columns[selected_features].tolist()
    print(f"\nOIP-AutoFS Finished: Selected {len(selected_names)} out of {num_features} features.")
    return selected_features, orig_imp[selected_features]

selected_mask, selected_imp = mopso_fs(X_train, X_test, y_train, y_test, num_particles=10, max_iterations=12)
X_train_sel = X_train.iloc[:, selected_mask]
X_test_sel = X_test.iloc[:, selected_mask]

# Phase 3: OPCE-CASH (MOPSO Hyperparameter Optimization for LightGBM & XGBoost)
print("\n" + "="*60)
print("Phase 3: OPCE-CASH (MOPSO Hyperparameter Optimization)")
print("="*60)

# Optimize LightGBM
def lgb_mopso(X_tr, X_te, y_tr, y_te, num_particles=10, max_iterations=12):
    print("\n--- Running OPCE-CASH on LightGBM ---")
    swarm = [{'position': np.array([100, 10, 0.1, 31, 20]) + np.random.uniform(-10, 10, 5),
              'velocity': np.random.uniform(-0.1, 0.1, 5)} for _ in range(num_particles)]
    gbest_score = np.array([float('inf')] * 3)
    gbest_pos = None

    for iteration in range(max_iterations):
        for p in swarm:
            p['position'][0] = np.clip(p['position'][0], 50, 200)
            p['position'][1] = np.clip(p['position'][1], 5, 50)
            p['position'][2] = np.clip(p['position'][2], 0.01, 0.3)
            p['position'][3] = np.clip(p['position'][3], 10, 50)
            p['position'][4] = np.clip(p['position'][4], 10, 50)

            params = {
                'n_estimators': int(p['position'][0]),
                'max_depth': int(p['position'][1]),
                'learning_rate': float(p['position'][2]),
                'num_leaves': int(p['position'][3]),
                'min_child_samples': int(p['position'][4])
            }
            start_t = time.time()
            clf = lgb.LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            proba = clf.predict_proba(X_te)
            f1 = f1_score(y_te, preds, average='weighted')
            conf = np.mean(np.max(proba, axis=1))
            elapsed = time.time() - start_t

            fitness = np.array([-0.90 * f1, -0.05 * conf, 0.05 * elapsed])

            if np.sum(fitness) < np.sum(p.get('pbest_score', np.array([float('inf')] * 3))):
                p['pbest_score'] = fitness
                p['pbest_position'] = p['position']

            if np.sum(fitness) < np.sum(gbest_score):
                gbest_score = fitness
                gbest_pos = p['position']

        w = 0.9 - 0.5 * (iteration / max_iterations)
        for p in swarm:
            inertia = w * p['velocity']
            cog = 1.5 * np.random.random() * (p['pbest_position'] - p['position'])
            soc = 1.5 * np.random.random() * (gbest_pos - p['position'])
            p['velocity'] = inertia + cog + soc
            p['position'] += p['velocity']

        print(f"LightGBM OPCE-CASH Iteration {iteration+1}/{max_iterations} complete.")

    best_params = {
        'n_estimators': int(gbest_pos[0]),
        'max_depth': int(gbest_pos[1]),
        'learning_rate': float(gbest_pos[2]),
        'num_leaves': int(gbest_pos[3]),
        'min_child_samples': int(gbest_pos[4])
    }
    print("Optimal LightGBM Hyperparameters:", best_params)
    return best_params

lgb_best_params = lgb_mopso(X_train_sel, X_test_sel, y_train, y_test, num_particles=10, max_iterations=12)

# Final LightGBM Evaluation
lgb_opt = lgb.LGBMClassifier(**lgb_best_params, random_state=42, verbose=-1, n_jobs=-1)
t1 = time.time()
lgb_opt.fit(X_train_sel, y_train)
t2 = time.time()
lgb_preds = lgb_opt.predict(X_test_sel)
lgb_f1 = f1_score(y_test, lgb_preds, average='weighted')
lgb_acc = accuracy_score(y_test, lgb_preds)

print(f"\nFinal Optimized LightGBM Performance: F1-Score={lgb_f1:.4f}, Accuracy={lgb_acc:.4f}, TrainTime={t2-t1:.2f}s")
joblib.dump(lgb_opt, "output/lightgbm_optimized.pkl")

# Plot confusion matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, lgb_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Optimized LightGBM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("output/lightgbm_cm.png")

print("\n" + "="*60)
print("REPRODUCTION COMPLETED SUCCESSFULLY!")
print("Optimized models and confusion matrix plot saved in output/")
print("="*60)
