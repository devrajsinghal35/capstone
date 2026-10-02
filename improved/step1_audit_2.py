import numpy as np
import time
from sklearn.metrics import brier_score_loss, log_loss

# Re-implement ECE from scratch
def compute_ece_scratch(y_true, y_proba, n_bins=15):
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

# Test with synthetic miscalibrated data
np.random.seed(42)
y_true_syn = np.random.randint(0, 2, 1000)
# Make perfectly confident but 50% wrong probabilities
y_proba_syn = np.zeros((1000, 2))
y_proba_syn[:, 0] = np.where(y_true_syn == 0, 1.0, 0.0)
# Miscalibrate by swapping some
swap_idx = np.random.choice(1000, 300, replace=False)
y_proba_syn[swap_idx, 0] = 1 - y_proba_syn[swap_idx, 0]
y_proba_syn[:, 1] = 1 - y_proba_syn[:, 0]

print("Synthetic ECE (should be ~0.3):", compute_ece_scratch(y_true_syn, y_proba_syn))

# True single-sample latency
def measure_latency(model, X_test):
    # Warmup
    for i in range(10):
        model.predict_proba(X_test.iloc[[i]])
    
    latencies = []
    # Test on 1000 samples to save time in audit
    for i in range(min(1000, len(X_test))):
        sample = X_test.iloc[[i]]
        t0 = time.time()
        model.predict_proba(sample)
        t1 = time.time()
        latencies.append((t1 - t0) * 1000)
        
    return np.median(latencies), np.percentile(latencies, 95)
