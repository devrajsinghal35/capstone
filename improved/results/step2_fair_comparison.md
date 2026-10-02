# Step 2: Fair Comparison (10 Seeds, Leak-Free)

Does oversampling help compared to native class weighting?

| Model    | Method                | Macro_F1        | Weighted_F1     | Accuracy        | MCC             | Bal_Acc         | Train_Time   |
|:---------|:----------------------|:----------------|:----------------|:----------------|:----------------|:----------------|:-------------|
| LightGBM | Buggy_SMOTE_ADASYN    | 0.9650 ± 0.0291 | 0.9961 ± 0.0007 | 0.9961 ± 0.0007 | 0.9914 ± 0.0016 | 0.9559 ± 0.0372 | 5.86s        |
| LightGBM | Class_Weight_Balanced | 0.9624 ± 0.0300 | 0.9961 ± 0.0010 | 0.9962 ± 0.0009 | 0.9916 ± 0.0020 | 0.9521 ± 0.0385 | 5.61s        |
| LightGBM | Default               | 0.6244 ± 0.3592 | 0.8625 ± 0.1523 | 0.8808 ± 0.1296 | 0.6993 ± 0.3473 | 0.6406 ± 0.3490 | 4.88s        |
| LightGBM | Fixed_SMOTE_ADASYN    | 0.9632 ± 0.0298 | 0.9958 ± 0.0009 | 0.9959 ± 0.0008 | 0.9909 ± 0.0018 | 0.9539 ± 0.0384 | 5.80s        |
| LightGBM | SMOTE_Only            | 0.9629 ± 0.0299 | 0.9954 ± 0.0009 | 0.9955 ± 0.0009 | 0.9900 ± 0.0019 | 0.9564 ± 0.0381 | 7.79s        |
| XGBoost  | Buggy_SMOTE_ADASYN    | 0.9644 ± 0.0295 | 0.9959 ± 0.0009 | 0.9960 ± 0.0008 | 0.9910 ± 0.0018 | 0.9568 ± 0.0380 | 2.63s        |
| XGBoost  | Class_Weight_Balanced | 0.9627 ± 0.0289 | 0.9958 ± 0.0007 | 0.9959 ± 0.0007 | 0.9908 ± 0.0014 | 0.9581 ± 0.0374 | 2.24s        |
| XGBoost  | Default               | 0.9624 ± 0.0303 | 0.9958 ± 0.0008 | 0.9959 ± 0.0007 | 0.9909 ± 0.0016 | 0.9489 ± 0.0374 | 2.35s        |
| XGBoost  | Fixed_SMOTE_ADASYN    | 0.9649 ± 0.0296 | 0.9959 ± 0.0006 | 0.9959 ± 0.0005 | 0.9910 ± 0.0012 | 0.9559 ± 0.0370 | 2.66s        |
| XGBoost  | SMOTE_Only            | 0.9623 ± 0.0298 | 0.9949 ± 0.0011 | 0.9949 ± 0.0010 | 0.9887 ± 0.0022 | 0.9598 ± 0.0381 | 5.44s        |

## Verification of Paper's Claims
- **Test Set Used During Tuning**: `fast_reproduce.py` lines 198-199 explicitly call `clf.fit(X_tr, y_tr)` and `preds = clf.predict(X_te)` directly inside `lgb_mopso`, where `X_te` is the global `X_test`.
- **Linear Scalarization instead of Pareto**: `fast_reproduce.py` line 205 calculates fitness as `np.array([-0.90 * f1, -0.05 * conf, 0.05 * elapsed])` and lines 207-208 just sum it: `if np.sum(fitness) < np.sum(...)`.
- **Latency measured as Training Time**: `fast_reproduce.py` lines 196-203 measures `start_t = time.time()`, trains the model `clf.fit(X_tr, y_tr)`, tests it, and then sets `elapsed = time.time() - start_t`. This mixes training time and batch testing time, not single-sample latency.
