# Comprehensive Methodology Report: MOO-AutoML IDS

This report details a step-by-step audit, correction, and enhancement of the original framework proposed by Yang & Shami (2025). It explicitly lists the methodological flaws found in the original code (the "Before"), the steps taken to fix them (the "After"), and the measurable metrics proving the validity of our improved pipeline.

---

## Step 1: Auditing the Baseline & Fixing Critical Flaws

### 1.1 Data Leakage
* **Before (Original Paper)**: The CICIDS2017 dataset subset contained 2,675 exact duplicate rows (~10% of the data). The original authors applied a random `train_test_split` on this raw data, meaning identical network traffic samples appeared in both the training set and the test set, artificially inflating their model's accuracy. Furthermore, hyperparameters were tuned by evaluating directly on the Test Set (`X_test`).
* **After (Our Improvement)**: We explicitly dropped all duplicate rows, reducing the dataset from 26,800 to 24,125 unique rows. We strictly isolated the Test Set. All hyperparameter tuning and multi-objective optimization (MOO) is now performed on a dedicated Validation Set.
* **Measurable Metric**: Test Set overlap dropped from **~10% (Leaked)** to **0% (Strictly Isolated)**.

### 1.2 The Class Imbalance Trap
* **Before**: The original authors used a computationally heavy `SMOTE + ADASYN` pipeline to artificially balance minority classes. 
* **After**: When we stripped away the data leakage and the SMOTE pipeline, a default LightGBM model plummeted to **71.10% Accuracy** and **12.95% Macro-F1**. The root cause was LightGBM's default `min_child_samples=20`, which suppressed leaf formation for extreme minority classes (e.g., Class 4 had only 29 training samples). By lowering this threshold or passing `class_weight='balanced'`, performance immediately recovered.
* **Measurable Metric**: Corrected Baseline Macro-F1 improved from **12.95%** to **97.04%** simply by adjusting native tree parameters for imbalanced data.

### 1.3 Metric & Latency Correction
* **Before**: The original paper reported an "ECE of 0.00%" using a flawed metric implementation that misaligned histogram bins. They also reported an impossible inference latency of `0.0033 ms` by measuring batch-training time rather than single-sample inference time.
* **After**: We implemented Expected Calibration Error (ECE) strictly from scratch using 15 equal-width bins. We also built a loop to measure true single-sample latency (`model.predict_proba(single_row)`) over 10,000 independent calls to simulate edge-device processing.

---

## Step 2: The Fair Comparison (Proving Oversampling is Unnecessary)

To determine if the original paper's slow `SMOTE+ADASYN` pipeline was actually necessary, we ran a fair comparison evaluating 10 random seeds on the leak-free, deduplicated dataset.

### **Before (Original Methodology)**
The original paper claimed that interpolating synthetic data (SMOTE+ADASYN) was essential for achieving high detection rates on minority attacks.

### **After (Our Evaluation)**
We evaluated native cost-sensitive learning (`class_weight='balanced'`) against both the original Buggy SMOTE pipeline and a Fixed SMOTE pipeline. 

| Model (LightGBM) | Method | Macro-F1 | Weighted-F1 | Train Time |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | Default | 62.44% ± 35.9 | 86.25% ± 15.2 | 4.88s |
| **Ours** | **Class_Weight_Balanced** | **96.24% ± 3.0** | **99.61% ± 0.1** | **5.61s** |
| Original | Buggy_SMOTE_ADASYN | 96.50% ± 2.9 | 99.61% ± 0.0 | 5.86s |
| Theoretical | Fixed_SMOTE_ADASYN | 96.32% ± 2.9 | 99.58% ± 0.0 | 5.80s |
| Traditional | SMOTE_Only | 96.29% ± 2.9 | 99.54% ± 0.0 | 7.79s |

* **Measurable Metric**: Native class weighting achieves a **99.61% Weighted-F1** and **96.24% Macro-F1**, which is statistically identical to the SMOTE pipelines. 
* **Conclusion**: Oversampling is entirely unnecessary for tree-based models on this dataset. Relying on native class weights removes the heavy computational overhead of synthetic data generation.

---

## Step 3: The Improved True MOO Pipeline (NSGA-II)

### **Before (Original Paper)**
The original paper claimed to use a Multi-Objective Particle Swarm Optimization (MOPSO) algorithm to balance Accuracy, Latency, and Confidence. However, their code explicitly used **linear scalarization** (`fitness = -0.90 * f1 - 0.05 * conf + 0.05 * elapsed`), which merges all objectives into a single number. This is not true MOO and fails to produce a Pareto front of trade-offs for network administrators.

### **After (Our Improvement)**
We replaced the scalarized MOPSO with a genuine multi-objective optimization engine using Optuna's **NSGA-II** algorithm.
1. **Search Space**: 50 multi-objective trials over `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, and `min_child_samples`.
2. **Objectives (Minimize)**: Macro-F1 Error, ECE (Calibration Error), and True Single-Sample Latency.
3. **Selection Rule**: Before touching the Test Set, we selected the model from the Validation Pareto Front that minimized Error while keeping Single-Sample Latency strictly under a 1.0 ms budget.

### **Final Measurable Metrics (On Hold-Out Test Set)**
The selected model was evaluated exactly once on the isolated test set:
* **Final Test Macro-F1**: **99.29%** (Proving exceptional detection of rare minority attacks without data leakage).
* **Final Test ECE**: **0.0023 (0.23%)** (Proving near-perfect probabilistic calibration).
* **Final Single-Sample Latency (Median)**: **0.5789 ms** (Proving the model can process over 1,700 independent packets per second on a single CPU thread, making it highly viable for IoT edge deployment).

---

## Conclusion
By fixing critical data leakage, replacing unnecessary and flawed oversampling with native cost-sensitive learning, and upgrading a fake scalarized MOO into a true NSGA-II Pareto search, we have transformed a flawed methodology into a mathematically rigorous, mathematically pure, and edge-deployable Cybersecurity IDS framework.
