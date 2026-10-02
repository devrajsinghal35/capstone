import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
import os

df = pd.read_csv("/Users/devrajsinghal/Desktop/capstone/Data/CICIDS2017_sample_km.csv")
print("Duplicates in dataset:", df.duplicated().sum())

y = df['Label']
counter = Counter(y)
print("Original class distribution:", counter)
target = int(sum(counter.values()) / len(counter) / 2)
minority_classes = {k: target for k, v in counter.items() if v < target}
print("Minority targets:", minority_classes)

try:
    X = df.drop('Label', axis=1)
    smote = SMOTE(sampling_strategy=minority_classes, k_neighbors=1, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print("After SMOTE distribution:", Counter(y_res))
    
    adasyn = ADASYN(sampling_strategy=minority_classes, n_neighbors=1, random_state=42)
    X_res2, y_res2 = adasyn.fit_resample(X_res, y_res)
    print("After ADASYN distribution:", Counter(y_res2))
except Exception as e:
    print("Exception during ADASYN:", str(e))
