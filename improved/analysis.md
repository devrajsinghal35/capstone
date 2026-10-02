# Phase 1 & 2: Understanding and Audit Analysis

## Phase 1: Understanding
### Summary of the Base Paper
* **Problem**: Need for autonomous, scalable, and efficient IDSs for resource-constrained environments like IoT.
* **Datasets**: Subsets of CICIDS2017 (26,800 samples) and IoTID20 (31,289 samples).
* **Methodology**: MOO-AutoML IDS Framework comprising:
  * **AutoDP**: Automated normalization (Shapiro-Wilk decides Z-score vs Min-Max) and hybrid data balancing (SMOTE + ADASYN).
  * **OIP-AutoFS**: MOPSO-based feature selection optimizing Information Gain (IG) and percentage of features.
  * **OPCE-CASH**: MOPSO-based hyperparameter tuning and model selection (XGBoost/LightGBM) optimizing F1-score, confidence, and sigmoid-normalized latency.
* **Reported Results**: Extremely high performance. Full pipeline on CICIDS2017 with XGBoost reported 99.773% F1.

## Phase 2: Audit and Identified Flaws (The Gaps)
Through code review and execution of audit scripts, we discovered severe methodological flaws in the original paper's implementation:

1. **Massive Data Leakage (Hyperparameter Optimization on Test Set)**:
   In `run_experiment.py` (OPCE-CASH step), the `objective_function` evaluates the MOPSO fitness directly on `X_test` and `y_test`. The exact same test set is then used for the final model evaluation (Line 739). This means the model's hyperparameters are explicitly tuned to maximize performance on the test set, invalidating the reported results.
2. **Data Leakage (Duplicates Across Train/Test)**:
   The dataset contains a significant number of duplicate rows (2,675 duplicates in the CICIDS2017 26,800 sample subset). Because the `train_test_split` is random and performed without deduplication, identical samples appear in both the training and test sets, inflating the test performance.
3. **Flawed Hybrid SMOTE+ADASYN Implementation**:
   Algorithm 1 claims SMOTE generates 50% of synthetic samples and ADASYN generates the other 50%. However, the code sets a target size (`minority_classes = target_samples`) and runs SMOTE. SMOTE upsamples the classes exactly to `target_samples`. When ADASYN is subsequently called with the same `target_samples` threshold, it observes that the classes already have the required number of samples and generates **zero** samples. The "hybrid" approach is functionally just SMOTE.
4. **Flawed Auto-Normalization**:
   The `Auto_Normalization` function applies the Shapiro-Wilk test to a flattened array of the first 1000 values across *all numeric features combined*. Mixing different feature distributions guarantees the rejection of normality, meaning Min-Max is always selected. Furthermore, tree-based models (XGBoost/LightGBM) are scale-invariant, making normalization redundant and computationally wasteful.
5. **Misrepresented Latency Objective in OPCE-CASH**:
   The paper claims to use "sigmoid-normalized latency" for the MOO efficiency objective. The code actually uses raw elapsed training time in seconds (not per-sample inference latency) multiplied by a static weight (`elapsed_time * time_weight`). It is never passed through a sigmoid function.
6. **Fake Multi-Objective Optimization (Scalarization)**:
   Both OIP-AutoFS and OPCE-CASH claim to perform Multi-Objective Optimization (MOPSO). However, the code uses linear scalarization (e.g., `score = f_w * norm_imp - p_w * feat_pct`) to convert multiple objectives into a single fitness score. No Pareto fronts or non-dominated sorting are actually used.
7. **Overconfidence Bias**:
   OPCE-CASH maximizes average predicted confidence. This forces the model to be overconfident (pushing probabilities to 1.0) rather than well-calibrated, degrading Expected Calibration Error (ECE) despite claims of improving reliability.

## Proposed Improvements for Our Original Paper
Ranked by impact and feasibility:
1. **Fix Leakage and Evaluation (Highest Impact/Feasibility)**: Remove duplicates before splitting. Perform hyperparameter tuning within proper nested CV or a separate validation set, completely isolating the test set. Use multi-seed evaluation (e.g., 10 seeds) with statistical significance testing.
2. **True Multi-Objective Optimization (High Impact/Feasibility)**: Replace the flawed scalarized MOPSO with Optuna's TPE for multi-objective optimization (NSGA-II under the hood), generating actual Pareto fronts for Accuracy vs. Inference Latency vs. Calibration (ECE).
3. **Real Efficiency (Medium Impact/Feasibility)**: Optimize for single-sample *inference* latency (simulating edge constraints), rather than raw batch training time.
4. **Proper Calibration (Medium Impact/Feasibility)**: Optimize for proper scoring rules (Brier score or log-loss) or ECE rather than raw confidence to ensure true reliability on the edge.
5. **Ablate Normalization & Oversampling (High Feasibility)**: Show that for tree models, removing the computationally heavy normalization and complex oversampling (or replacing with class weights) maintains performance while drastically reducing pipeline latency.

**Next Steps**: Await user approval on this audit and plan before implementing Phase 3.
