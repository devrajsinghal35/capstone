# Step 1: Sanity-Check Audit Findings

## 1. Baseline Audit
* **Symptom**: Corrected baseline (default LightGBM on deduplicated CICIDS2017) scored 71.10% accuracy / 61.05% F1, far below the expected ~99%.
* **Investigation**: We checked label mapping, NaN/inf handling (no infs found, NaNs cleaned), and deduplication impact (classes were reduced but proportional). Comparing XGBoost and RandomForest on the exact same deduplicated split yielded 99.65% and 99.30% accuracy respectively.
* **Root Cause**: The fault lies entirely in the LightGBM default configuration for highly imbalanced data. LightGBM defaults to `min_child_samples=20` (minimum data in leaf). In our deduplicated split, minority classes like Class 4 (Infiltration) only have 29 samples in the training set, and Class 2 has 70. LightGBM's histogram-based splitting suppresses leaf formation for these tiny, imbalanced classes under the default constraints, causing the model to collapse to majority-class predictions.
* **Fix**: Reducing `min_child_samples` (e.g., to 5) or using `class_weight='balanced'` immediately restores LightGBM's performance to 99.65% Accuracy and 97.04% Macro-F1. We will use `class_weight='balanced'` and tune `min_child_samples` appropriately in the fair comparison.

## 2. ECE Audit
* **Symptom**: We incorrectly reported ECE = 0.000% for both the 71% accurate model and the 99% accurate model.
* **Root Cause**: The previous `compute_ece` function used `sklearn.calibration.calibration_curve`. `calibration_curve` automatically drops empty bins. We then computed bin weights using `np.histogram(bins=10)`, which *keeps* empty bins. Slicing `bin_counts[:len(prob_true)]` completely misaligned the weights with the bins, zeroing out the ECE.
* **Fix**: We have re-implemented ECE from scratch using 15 strict equal-width bins (`np.linspace(0, 1, 16)`). Empty bins are explicitly skipped, and weights are correctly assigned. The unit test on synthetic miscalibrated data confirms it now correctly outputs ~0.3 ECE. We will also include Brier score and log-loss metrics.

## 3. Metric Audit
* **Symptom**: Accuracy and F1 were virtually identical in previous outputs.
* **Root Cause**: We were using `f1_score(average='weighted')`. In highly imbalanced datasets, weighted F1 is heavily dominated by the majority class, essentially mirroring Accuracy and Micro-F1.
* **Fix**: We will replace Weighted F1 with **Macro-F1**, which averages the F1 scores of each class equally, heavily penalizing models that fail on minority classes. We will also report MCC, balanced accuracy, and per-class precision/recall.

## 4. Latency Audit
* **Symptom**: The reported 0.0033 ms/sample was physically unrealistic for single-row inference.
* **Root Cause**: It was calculated as batch-amortized latency (`(t2 - t1) / len(X_test)`), which takes advantage of vectorized operations and CPU cache on large batches.
* **Fix**: We wrote a loop to measure true single-sample latency (`model.predict_proba(single_row)`) over 10,000 calls. We will report the Median and P95 true single-sample latencies independently of batch-amortized latencies.

## 5. Leakage Audit
* **Symptom**: Data leakage artificially inflating prior results.
* **Root Cause**: 2,675 exact duplicate rows in the CICIDS2017 subset.
* **Fix**: Exact duplicates are stripped *before* splitting. We will implement a strict cross-check assertion `assert len(set(X_train.index).intersection(set(X_test.index))) == 0` to guarantee no index leakage. All tuning will explicitly target a validation fold isolated from the test set.

**Action**: Pausing here per instructions. Awaiting review of these diagnostics before proceeding to Step 2 (Fair Comparison).
