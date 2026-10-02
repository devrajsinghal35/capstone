# Step 3: Improved MOO Pipeline (NSGA-II)

## Pareto Front Model Selection
Optuna found 7 models on the Pareto Front across (Error, ECE, Latency).
We selected the model strictly based on the Validation split with the rule: Minimize Error where Latency < 1.0 ms.

### Chosen Hyperparameters:
```json
{'n_estimators': 64, 'max_depth': 14, 'learning_rate': 0.21380932890222198, 'num_leaves': 35, 'min_child_samples': 20, 'class_weight': 'balanced', 'random_state': 42, 'verbose': -1, 'n_jobs': -1}
```

## Final Hold-out Test Set Performance
- **Macro-F1**: 0.9929
- **ECE**: 0.0023
- **Single-Sample Latency (Median)**: 0.5789 ms
