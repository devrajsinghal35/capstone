#!/usr/bin/env python
# coding: utf-8

# # Toward Autonomous and Efficient Cybersecurity: A Multi Objective AutoML based Intrusion Detection System 
# This is the code for the paper entitled "[**Toward Autonomous and Efficient Cybersecurity: A Multi Objective AutoML based Intrusion Detection System**](https://ieeexplore.ieee.org/document/11240569/)" published in IEEE Transactions on Machine Learning in Communications and Networking (TMLCN).  
# Authors: Li Yang (liyanghart@gmail.com) and Abdallah Shami  
# 
# If you find this repository useful in your research, please cite this article as:  
# L. Yang and A. Shami, “Towards Autonomous and Efficient Cybersecurity: A Multi Objective AutoML based Intrusion Detection System,” _IEEE Transactions on Machine Learning in Communications and Networking_, pp. 1–21, 2025, doi: [10.1109/TMLCN.2025.3631379](https://ieeexplore.ieee.org/document/11240569/)   
# 
# ```
# @ARTICLE{11240569,
#   author={Yang, Li and Shami, Abdallah},
#   journal={IEEE Transactions on Machine Learning in Communications and Networking}, 
#   title={Towards Autonomous and Efficient Cybersecurity: A Multi-Objective AutoML-based Intrusion Detection System}, 
#   year={2025},
#   pages={1-21},
#   keywords={Computer security;Automated machine learning;Optimization;Internet of Things;Intrusion detection;Feature extraction;Data models;Data analysis;Benchmark testing;Adaptation models;Network Automation;AutoML;Multi-Objective Optimization;Cybersecurity;Intrusion Detection System;IoT},
#   doi={10.1109/TMLCN.2025.3631379}}
# ```
# 

# ## Import libraries

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
import joblib
import os
import time


# ## Dataset 1: CICIDS2017
# The CICIDS2017 dataset is publicly available at: https://www.unb.ca/cic/datasets/ids-2017.html  
# 
# Due to the large size of this dataset and the file size limit of GitHub, the sampled subset of CICIDS2017 is used. The subsets are in the "Data" folder.  PS: The results might be different from the paper due to the size difference of the dataset.
# 
# The Canadian Institute for Cybersecurity Intrusion Detection System 2017 (CICIDS2017) dataset has the most updated network threats. The CICIDS2017 dataset is close to real-world network data since it has a large amount of network traffic data, a variety of network features, various types of attacks, and highly imbalanced classes.
# 
# If you want to use this code on other datasets, just change the dataset name and follow the same steps. The models in this code are generic models that can be used in any intrusion detection/network traffic datasets.

# In[3]:


# Read the dataset
df = pd.read_csv("Data/CICIDS2017_sample_km.csv")


# In[4]:


# Display the data
df


# In[5]:


df.Label.value_counts()


# **Corresponding Attack Types:**  
# 0 BENIGN &emsp; 18225  
# 3 DoS        &emsp;   &emsp;   3042  
# 6 WebAttack    &emsp;      2180  
# 1 Bot        &emsp;  &emsp;      1966    
# 5 PortScan  &emsp;       1255  
# 2 BruteForce  &emsp;      96  
# 4 Infiltration  &emsp;       36  

# ## 1. Automated Data Pre-Processing

# ### Automated normalization
# Normalize the range of features to a similar scale to improve data quality

# In[6]:


from scipy.stats import shapiro
def Auto_Normalization(df):
    stat, p = shapiro(df)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    numeric_features = df.drop(['Label'],axis = 1).dtypes[df.dtypes != 'object'].index

    # The selection strategy is based on the following article: 
    # https://medium.com/@kumarvaishnav17/standardization-vs-normalization-in-machine-learning-3e132a19c8bf
    # Check if the data distribution follows a Gaussian/normal distribution
    # If so, select the Z-score normalization method; otherwise, select the min-max normalization
    # Details are in the paper
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
        df[numeric_features] = df[numeric_features].apply(
            lambda x: (x - x.mean()) / (x.std()))
        print('Z-score normalization is automatically chosen and used')
    else:
        print('Sample does not look Gaussian (reject H0)')
        df[numeric_features] = df[numeric_features].apply(
            lambda x: (x - x.min()) / (x.max()-x.min()))
        print('Min-max normalization is automatically chosen and used')
    return df
df = Auto_Normalization(df)


# In[7]:


# Address missing and infinite values
if df.isnull().values.any() or np.isinf(df).values.any(): # if there is any empty or infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace = True)  # Replace empty values with zeros


# ### Split train set and test set

# In[8]:


# Split the dataset into training and testing
X = df.drop(['Label'],axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0) #shuffle=False


# ### Machine learning model training

# ### Training base learners (for comparison purposes): 
# decision tree, random forest, extra trees, XGBoost, and LightGBM
# 

# #### DT

# In[9]:


get_ipython().run_cell_magic('time', '', '# Train the Decision Tree classifier\ndecision_tree = DecisionTreeClassifier(random_state=42)\n\n# model training\nt1 = time.time()\ndecision_tree.fit(X_train, y_train)\nt2 = time.time()\nprint("Training time: ", t2-t1, "s")\n\nt3 = time.time()\ny_pred = decision_tree.predict(X_test)\nt4 = time.time()\nprint("Prediction time per sample: ", (t4-t3)/len(X_test)*1000, "ms")\n\n# Print metrics\nprint(classification_report(y_test, y_pred))\nprint("Accuracy of Decision Tree: " + str(accuracy_score(y_test, y_pred)))\nprint("Precision of Decision Tree: " + str(precision_score(y_test, y_pred, average=\'weighted\')))\nprint("Recall of Decision Tree: " + str(recall_score(y_test, y_pred, average=\'weighted\')))\nprint("Average F1 of Decision Tree: " + str(f1_score(y_test, y_pred, average=\'weighted\')))\nprint("F1 of Decision Tree for each type of attack: " + str(f1_score(y_test, y_pred, average=None)))\ndecision_tree_f1 = f1_score(y_test, y_pred, average=None)\n\n# Plot the confusion matrix\ncm = confusion_matrix(y_test, y_pred)\nf, ax = plt.subplots(figsize=(5, 5))\nsns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)\nplt.xlabel("y_pred")\nplt.ylabel("y_true")\nplt.show()\n')


# #### RF

# In[10]:


get_ipython().run_cell_magic('time', '', '\n# Train the Random Forest classifier\nrf = RandomForestClassifier(random_state=42)\nt1 = time.time()\nrf.fit(X_train, y_train)\nt2 = time.time()\nprint("Training time: ", t2-t1, "s")\n\n# Make predictions\nt3 = time.time()\ny_pred = rf.predict(X_test)\nt4 = time.time()\nprint("Prediction time per sample: ", (t4-t3)/len(X_test)*1000, "ms")\n\n# Print metrics\nprint(classification_report(y_test, y_pred))\nprint("Accuracy of RF: " + str(accuracy_score(y_test, y_pred)))\nprint("Precision of RF: " + str(precision_score(y_test, y_pred, average=\'weighted\')))\nprint("Recall of RF: " + str(recall_score(y_test, y_pred, average=\'weighted\')))\nprint("Average F1 of RF: " + str(f1_score(y_test, y_pred, average=\'weighted\')))\nprint("F1 of RF for each type of attack: " + str(f1_score(y_test, y_pred, average=None)))\n\n# Store F1 scores\nrf_f1 = f1_score(y_test, y_pred, average=None)\n\n# Plot the confusion matrix\ncm = confusion_matrix(y_test, y_pred)\nf, ax = plt.subplots(figsize=(5, 5))\nsns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)\nplt.xlabel("Predicted")\nplt.ylabel("True")\nplt.show()\n')


# #### ET

# In[12]:


get_ipython().run_cell_magic('time', '', '# Extra Trees training\net = ExtraTreesClassifier(random_state=42, n_jobs=-1)\nt1 = time.time()\net.fit(X_train, y_train)\nt2 = time.time()\nprint("Training time: ", t2 - t1, "s")\n\n# Make predictions\nt3 = time.time()\ny_pred = et.predict(X_test)\nt4 = time.time()\nprint("Prediction time per sample: ", (t4 - t3) / len(X_test) * 1000, "ms")\n\n# Print metrics\nprint(classification_report(y_test, y_pred))\nprint("Accuracy of ET: " + str(accuracy_score(y_test, y_pred)))\nprint("Precision of ET: " + str(precision_score(y_test, y_pred, average=\'weighted\')))\nprint("Recall of ET: " + str(recall_score(y_test, y_pred, average=\'weighted\')))\nprint("Average F1 of ET: " + str(f1_score(y_test, y_pred, average=\'weighted\')))\nprint("F1 of ET for each type of attack: " + str(f1_score(y_test, y_pred, average=None)))\n\n# Store F1 scores\net_f1 = f1_score(y_test, y_pred, average=None)\n\n# Plot the confusion matrix (unchanged)\ncm = confusion_matrix(y_test, y_pred)\nf, ax = plt.subplots(figsize=(5, 5))\nsns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)\nplt.xlabel("Predicted")\nplt.ylabel("True")\nplt.show()\n')


# #### XGBoost

# In[13]:


get_ipython().run_cell_magic('time', '', '# Train the XGBoost algorithm\n\nxg = xgb.XGBClassifier(random_state=42, objective="multi:softprob", eval_metric="mlogloss")\n\n# X_train_x = X_train.values\n# X_test_x = X_test.values\n\nt1 = time.time()\nxg.fit(X_train, y_train)\nt2 = time.time()\nprint("Training time: ", t2-t1, "s")\n\nt3 = time.time()\ny_pred = xg.predict(X_test)\nt4 = time.time()\nprint("Prediction time per sample: ", (t4-t3)/len(X_test)*1000, "ms")\n\nprint(classification_report(y_test,y_pred))\nprint("Accuracy of XGBoost: "+ str(accuracy_score(y_test, y_pred)))\nprint("Precision of XGBoost: "+ str(precision_score(y_test, y_pred, average=\'weighted\')))\nprint("Recall of XGBoost: "+ str(recall_score(y_test, y_pred, average=\'weighted\')))\nprint("Average F1 of XGBoost: "+ str(f1_score(y_test, y_pred, average=\'weighted\')))\nprint("F1 of XGBoost for each type of attack: "+ str(f1_score(y_test, y_pred, average=None)))\nxg_f1=f1_score(y_test, y_pred, average=None)\n\n# Plot the confusion matrix\ncm=confusion_matrix(y_test,y_pred)\nf,ax=plt.subplots(figsize=(5,5))\nsns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)\nplt.xlabel("y_pred")\nplt.ylabel("y_true")\nplt.show()\n')


# #### LightGBM

# In[14]:


get_ipython().run_cell_magic('time', '', '# Train the LightGBM algorithm\nlg = lgb.LGBMClassifier(random_state=42)\n\nt1 = time.time()\nlg.fit(X_train, y_train)\nt2 = time.time()\nprint("Training time: ", t2-t1, "s")\n\nt3 = time.time()\ny_pred = lg.predict(X_test)\nt4 = time.time()\nprint("Prediction time per sample: ", (t4-t3)/len(X_test)*1000, "ms")\n\nprint(classification_report(y_test,y_pred))\nprint("Accuracy of LightGBM: "+ str(accuracy_score(y_test, y_pred)))\nprint("Precision of LightGBM: "+ str(precision_score(y_test, y_pred, average=\'weighted\')))\nprint("Recall of LightGBM: "+ str(recall_score(y_test, y_pred, average=\'weighted\')))\nprint("Average F1 of LightGBM: "+ str(f1_score(y_test, y_pred, average=\'weighted\')))\nprint("F1 of LightGBM for each type of attack: "+ str(f1_score(y_test, y_pred, average=None)))\nlg_f1=f1_score(y_test, y_pred, average=None)\n\n# Plot the confusion matrix\ncm=confusion_matrix(y_test,y_pred)\nf,ax=plt.subplots(figsize=(5,5))\nsns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)\nplt.xlabel("y_pred")\nplt.ylabel("y_true")\nplt.show()\n')


# Initial model selection: XGBoost and LightGBM are the two best-performing models.

# ### Automated hybrid data balancing
# Automated hybrid data balancing approach integrating SMOTE and ADASYN oversampling techniques to address class imbalance issues.

# In[15]:


# Display the class distribution in the training set
pd.Series(y_train).value_counts()


# In[16]:


# Proposed hybrid data balancing approach
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter

# Class-imbalance detection
average_samples_per_class = sum(Counter(y_train).values()) / len(Counter(y_train)) # Average number of samples per class
target_samples = int(average_samples_per_class / 2) # Target number of samples (average number/2) for each minority class
minority_classes = {k: target_samples for k, v in Counter(y_train).items() if v < target_samples}

# Check if there are minority classes
if minority_classes:
    # Apply SMOTE for 50% of the required samples
    smote = SMOTE(n_jobs=-1, sampling_strategy=minority_classes)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Apply ADASYN for 50% of the required samples
    adasyn = ADASYN(n_jobs=-1, sampling_strategy=minority_classes)
    X_train, y_train = adasyn.fit_resample(X_train, y_train)

# Check the final class distribution
print(pd.Series(y_train).value_counts())


# In[17]:


get_ipython().run_cell_magic('time', '', '# LightGBM model performance after hybrid data balancing\nimport lightgbm as lgb\nlg = lgb.LGBMClassifier(random_state=42)\n\nt1 = time.time()\nlg.fit(X_train, y_train)\nt2 = time.time()\nprint("Training time: ", t2-t1, "s")\n\nt3 = time.time()\ny_pred = lg.predict(X_test)\nt4 = time.time()\nprint("Prediction time per sample: ", (t4-t3)/len(X_test)*1000, "ms")\n\nprint(classification_report(y_test,y_pred))\nprint("Accuracy of LightGBM: "+ str(accuracy_score(y_test, y_pred)))\nprint("Precision of LightGBM: "+ str(precision_score(y_test, y_pred, average=\'weighted\')))\nprint("Recall of LightGBM: "+ str(recall_score(y_test, y_pred, average=\'weighted\')))\nprint("Average F1 of LightGBM: "+ str(f1_score(y_test, y_pred, average=\'weighted\')))\nprint("F1 of LightGBM for each type of attack: "+ str(f1_score(y_test, y_pred, average=None)))\nlg_f1=f1_score(y_test, y_pred, average=None)\n\n# Plot the confusion matrix\ncm=confusion_matrix(y_test,y_pred)\nf,ax=plt.subplots(figsize=(5,5))\nsns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)\nplt.xlabel("y_pred")\nplt.ylabel("y_true")\nplt.show()\n')


# In[18]:


get_ipython().run_cell_magic('time', '', '# XGBoost model performance after hybrid data balancing\nimport xgboost as xgb\nxg = xgb.XGBClassifier(random_state=42, objective="multi:softprob", eval_metric="mlogloss")\n\n# X_train_x = X_train.values\n# X_test_x = X_test.values\n\nt1 = time.time()\nxg.fit(X_train, y_train)\nt2 = time.time()\nprint("Training time: ", t2-t1, "s")\n\nt3 = time.time()\ny_pred = xg.predict(X_test)\nt4 = time.time()\nprint("Prediction time per sample: ", (t4-t3)/len(X_test)*1000, "ms")\n\nprint(classification_report(y_test,y_pred))\nprint("Accuracy of XGBoost: "+ str(accuracy_score(y_test, y_pred)))\nprint("Precision of XGBoost: "+ str(precision_score(y_test, y_pred, average=\'weighted\')))\nprint("Recall of XGBoost: "+ str(recall_score(y_test, y_pred, average=\'weighted\')))\nprint("Average F1 of XGBoost: "+ str(f1_score(y_test, y_pred, average=\'weighted\')))\nprint("F1 of XGBoost for each type of attack: "+ str(f1_score(y_test, y_pred, average=None)))\nxg_f1=f1_score(y_test, y_pred, average=None)\n\n# Plot the confusion matrix\ncm=confusion_matrix(y_test,y_pred)\nf,ax=plt.subplots(figsize=(5,5))\nsns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)\nplt.xlabel("y_pred")\nplt.ylabel("y_true")\nplt.show()\n')


# ## 2. Automated Feature Selection
# Proposed Optimized Importance and Percentage-based Automated Feature Selection (OIP-AutoFS), which is driven by a Multi-Objective Particle Swarm Optimization (MOPSO) and feature importance. The main purpose of this AutoFS process is to optimize the accumulated importance and percentage of features, thus ensuring that the most significant features are used for model training. 

# In[ ]:


# Train a baseline LightGBM model on the full feature set to get total feature importance
def get_original_feature_importance(X_train, y_train):
    original_clf = lgb.LGBMClassifier()
    original_clf.fit(X_train, y_train)
    return original_clf.feature_importances_

# Define the objective function (Optimizing Feature Importance and Feature Percentage)
def objective_function(position, X_train, X_test, y_train, y_test, 
                       original_feature_importances, feature_importance_weight=0.9, 
                       feature_percentage_weight=0.1): # Weights for the two objectives, can be tuned
    mask = position > 0.5
    X_train_selected = X_train.iloc[:, mask]
    X_test_selected = X_test.iloc[:, mask]

    if X_train_selected.shape[1] == 0:  # Prevent empty feature selection
        return float('-inf'), 1.0  # Worst possible score

    # Train LightGBM model on selected features
    clf = lgb.LGBMClassifier()
    clf.fit(X_train_selected, y_train)

    # Extract the original importance values corresponding to selected features
    selected_feature_importance = original_feature_importances[mask]
    total_selected_importance = np.sum(selected_feature_importance)

    # Normalize feature importance score against the total original importance (range 0-1)
    total_feature_importance_normalized = total_selected_importance / np.sum(original_feature_importances) if np.sum(original_feature_importances) > 0 else 0

    # Compute the percentage of selected features
    num_selected_features = np.sum(mask)
    feature_percentage = num_selected_features / X_train.shape[1]

    # Objective function: Maximizing feature importance and minimizing feature percentage
    score = (
        feature_importance_weight * total_feature_importance_normalized - 
        feature_percentage_weight * feature_percentage
    )

    return score, feature_percentage

# Initialize swarm
def initialize_swarm(num_particles, num_features):
    return [{'position': np.random.rand(num_features), 'velocity': np.random.uniform(-0.1, 0.1, num_features)} for _ in range(num_particles)]

# Update velocity and position with adaptive inertia weight
def update_velocity_position(particle, pbest_position, gbest_position, iteration, max_iterations, 
                             w_max=0.9, w_min=0.4, c1=1.5, c2=1.5):

    # **Adaptive inertia weight**: Starts high, decreases over iterations for better convergence
    w = w_max - (w_max - w_min) * (iteration / max_iterations)

    inertia = w * particle['velocity']
    cognitive = c1 * np.random.random() * (pbest_position - particle['position'])
    social = c2 * np.random.random() * (gbest_position - particle['position'])

    inertia = w * particle['velocity']
    cognitive = c1 * np.random.random() * (pbest_position - particle['position'])
    social = c2 * np.random.random() * (gbest_position - particle['position'])
    new_velocity = inertia + cognitive + social
    new_position = particle['position'] + new_velocity

    # Apply bounds
    new_position = np.clip(new_position, 0, 1)
    new_velocity = np.clip(new_velocity, -0.1, 0.1)

    particle['velocity'] = new_velocity
    particle['position'] = new_position

# Main MOPSO function
def mopso(X_train, X_test, y_train, y_test, num_particles=10, max_iterations=20):
    num_features = X_train.shape[1]
    original_feature_importances = get_original_feature_importance(X_train, y_train)  # Get true total feature importance before selection

    swarm = initialize_swarm(num_particles, num_features)
    gbest_score = float('-inf')
    gbest_position = None

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")

        for particle in swarm:
            fitness = objective_function(particle['position'], X_train, X_test, y_train, y_test, original_feature_importances)

            # Update personal best
            if fitness[0] > particle.get('pbest_score', float('-inf')):
                particle['pbest_score'] = fitness[0]
                particle['pbest_position'] = particle['position']

            # Update global best
            if fitness[0] > gbest_score:
                gbest_score = fitness[0]
                gbest_position = particle['position']

        # Update velocity and position
        for particle in swarm:
            update_velocity_position(particle, particle['pbest_position'], gbest_position, iteration, max_iterations)

    # Get final selected features
    selected_features = gbest_position > 0.5
    selected_feature_names = X_train.columns[selected_features].tolist()
    print("\nSelected Features:", selected_feature_names)

    # Compute final selected feature importance
    selected_feature_importance = original_feature_importances[selected_features]
    final_total_selected_importance = np.sum(selected_feature_importance)

    # Normalize feature importance against the total original importance
    final_relative_feature_importance = final_total_selected_importance / np.sum(original_feature_importances) if np.sum(original_feature_importances) > 0 else 0

    # Compute final percentage of selected features
    final_percentage_selected = np.sum(selected_features) / X_train.shape[1]

    print("\nFinal Relative Accumulated Feature Importance Score (0-1):", final_relative_feature_importance)
    print("Final Percentage of Selected Features:", final_percentage_selected)


    return selected_features, gbest_position, selected_feature_importance


# In[27]:


selected_features, gbest_position, selected_feature_importance = mopso(X_train, X_test, y_train, y_test)


# In[28]:


# Create new datasets with selected features
X_train_selected = X_train.iloc[:, selected_features]
X_test_selected = X_test.iloc[:, selected_features]
len(X_train.columns[selected_features].tolist())


# In[29]:


# Plot the feature importance of the selected features
plt.rcParams.update({'font.size': 12})

# Create a DataFrame for easier plotting
features = pd.DataFrame({
    'Feature': X_train.columns[selected_features],
    'Importance': selected_feature_importance
})

# Normalize to relative importance within the selected feature set
imp_sum = features['Importance'].sum()
if imp_sum > 0:
    features['Importance'] = features['Importance'] / imp_sum
else:
    n = len(features)
    features['Importance'] = 1.0 / n if n > 0 else 0.0

# Sort features by importance
features = features.sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(20, 6))
scatter = plt.scatter(x='Feature', y='Importance', s=200, c='Importance', cmap='viridis', alpha=0.6, data=features)
plt.colorbar(scatter, label='Normalized Importance')
plt.xticks(rotation=45, ha='right')
plt.title('Selected Feature Importance of CICIDS2017 Dataset')
plt.ylabel('Normalized Feature Importance')
plt.grid(True)
plt.show()


# In[30]:


get_ipython().run_cell_magic('time', '', '# LightGBM model performance after AutoDP and AutoFS\nimport lightgbm as lgb\nlg = lgb.LGBMClassifier(random_state=42)\n\nt1 = time.time()\nlg.fit(X_train_selected, y_train)\nt2 = time.time()\nprint("Training time: ", t2-t1, "s")\n\nt3 = time.time()\ny_pred = lg.predict(X_test_selected)\nt4 = time.time()\nprint("Prediction time per sample: ", (t4-t3)/len(X_test)*1000, "ms")\n\nprint(classification_report(y_test,y_pred))\nprint("Accuracy of LightGBM: "+ str(accuracy_score(y_test, y_pred)))\nprint("Precision of LightGBM: "+ str(precision_score(y_test, y_pred, average=\'weighted\')))\nprint("Recall of LightGBM: "+ str(recall_score(y_test, y_pred, average=\'weighted\')))\nprint("Average F1 of LightGBM: "+ str(f1_score(y_test, y_pred, average=\'weighted\')))\nprint("F1 of LightGBM for each type of attack: "+ str(f1_score(y_test, y_pred, average=None)))\nlg_f1=f1_score(y_test, y_pred, average=None)\n\n# Plot the confusion matrix\ncm=confusion_matrix(y_test,y_pred)\nf,ax=plt.subplots(figsize=(5,5))\nsns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)\nplt.xlabel("y_pred")\nplt.ylabel("y_true")\nplt.show()\n')


# In[31]:


get_ipython().run_cell_magic('time', '', '# XGBoost model performance after AutoDP and AutoFS\nimport xgboost as xgb\nxg = xgb.XGBClassifier(random_state=42, objective="multi:softprob", eval_metric="mlogloss")\n\n# X_train_x = X_train.values\n# X_test_x = X_test.values\n\nt1 = time.time()\nxg.fit(X_train_selected, y_train)\nt2 = time.time()\nprint("Training time: ", t2-t1, "s")\n\nt3 = time.time()\ny_pred = xg.predict(X_test_selected)\nt4 = time.time()\nprint("Prediction time per sample: ", (t4-t3)/len(X_test)*1000, "ms")\n\nprint(classification_report(y_test,y_pred))\nprint("Accuracy of XGBoost: "+ str(accuracy_score(y_test, y_pred)))\nprint("Precision of XGBoost: "+ str(precision_score(y_test, y_pred, average=\'weighted\')))\nprint("Recall of XGBoost: "+ str(recall_score(y_test, y_pred, average=\'weighted\')))\nprint("Average F1 of XGBoost: "+ str(f1_score(y_test, y_pred, average=\'weighted\')))\nprint("F1 of XGBoost for each type of attack: "+ str(f1_score(y_test, y_pred, average=None)))\nxg_f1=f1_score(y_test, y_pred, average=None)\n\n# Plot the confusion matrix\ncm=confusion_matrix(y_test,y_pred)\nf,ax=plt.subplots(figsize=(5,5))\nsns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)\nplt.xlabel("y_pred")\nplt.ylabel("y_true")\nplt.show()\n')


# ## 3. Model Selection and Hyperparameter Optimization
# The automated model learning and optimization using the proposed OPCE-CASH method is the final phase, a critical step where XGBoost and LightGBM are automatically optimized and selected using the MOPSO model. This phase aims to optimize the F1-score (representing model effectiveness), confidence values (representing model reliability), and execution time (indicating model complexity). 

# ### Optimize the LightGBM model

# In[32]:


# Display the default hyperparameters of LightGBM
lg.get_params()


# In[33]:


# Write the MOPSO algorithm to optimize LightGBM hyperparameters

counter = 0  # Declare this at the top-level script
# Define the objective function (Optimizing F1 Score, Average Confidence, and Training Time)
def objective_function(position, X_train, X_test, y_train, y_test, f1_weight=0.90, confidence_weight=0.05, time_weight=0.05):
    global counter  # Declare counter as global to modify it
    counter += 1  # Increment counter

    start_time = time.time()

    # Apply hard constraints
    position[0] = np.clip(position[0], 50, 200)  # n_estimators
    position[1] = np.clip(position[1], 5, 100)    # max_depth
    position[2] = np.clip(position[2], 0.01, 0.3)  # learning_rate
    position[3] = np.clip(position[3], 10, 50)   # num_leaves
    position[4] = np.clip(position[4], 10, 50)   # min_child_samples

    hyperparams = {
        'n_estimators': int(position[0]),
        'max_depth': int(position[1]),
        'learning_rate': position[2],
        'num_leaves': int(position[3]),
        'min_child_samples': int(position[4])
    }

    print(f"Evaluation {counter}: Hyperparameters before training:", hyperparams)  # Debugging line

    clf = lgb.LGBMClassifier(**hyperparams, random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

    f1 = f1_score(y_test, predictions, average='weighted')
    negative_f1 = -f1 * f1_weight

    avg_confidence = np.mean(np.max(proba, axis=1))
    negative_avg_confidence = -avg_confidence * confidence_weight

    elapsed_time = time.time() - start_time
    weighted_elapsed_time = elapsed_time * time_weight

    return negative_f1, negative_avg_confidence, weighted_elapsed_time



# Initialize swarm
def initialize_swarm(num_particles):
    return [{'position': np.array([100, 10, 0.1, 31, 20]) + np.random.uniform(-10, 10, 5), 
             'velocity': np.random.uniform(-0.1, 0.1, 5)} for _ in range(num_particles)]

# Update velocity and position with adaptive inertia weight
def update_velocity_position(particle, pbest_position, gbest_position, iteration, max_iterations, 
                             w_max=0.9, w_min=0.4, c1=1.5, c2=1.5):

    # **Adaptive inertia weight**: Starts high, decreases over iterations for better convergence
    w = w_max - (w_max - w_min) * (iteration / max_iterations)

    inertia = w * particle['velocity']
    cognitive = c1 * np.random.random() * (pbest_position - particle['position'])
    social = c2 * np.random.random() * (gbest_position - particle['position'])

    new_velocity = inertia + cognitive + social
    new_position = particle['position'] + new_velocity

    # Apply constraints
    new_position[0] = np.clip(new_position[0], 50, 200)  # n_estimators
    new_position[1] = np.clip(new_position[1], 5, 100)    # max_depth
    new_position[2] = np.clip(new_position[2], 0.01, 0.3)  # learning_rate
    new_position[3] = np.clip(new_position[3], 10, 50)   # num_leaves
    new_position[4] = np.clip(new_position[4], 10, 50)   # min_child_samples

    particle['velocity'] = new_velocity
    particle['position'] = new_position


# Main MOPSO function
def mopso(X_train, X_test, y_train, y_test, num_particles=20, max_iterations=20):
    swarm = initialize_swarm(num_particles)
    gbest_score = np.array([float('inf')] * 3)
    gbest_position = None

    for iteration in range(max_iterations):
        for particle in swarm:
            fitness = np.array(objective_function(particle['position'], X_train, X_test, y_train, y_test))

            if np.sum(fitness) < np.sum(particle.get('pbest_score', np.array([float('inf')] * 3))):
                particle['pbest_score'] = fitness
                particle['pbest_position'] = particle['position']

            if np.sum(fitness) < np.sum(gbest_score):
                gbest_score = fitness
                gbest_position = particle['position']

        for particle in swarm:
            update_velocity_position(particle, particle['pbest_position'], gbest_position, iteration, max_iterations)


    print(f"Optimal hyperparameters are n_estimators: {int(gbest_position[0])}, max_depth: {int(gbest_position[1])}, learning_rate: {gbest_position[2]}, num_leaves: {int(gbest_position[3])}, min_child_samples: {int(gbest_position[4])}")

    return {
        'n_estimators': int(gbest_position[0]),
        'max_depth': int(gbest_position[1]),
        'learning_rate': gbest_position[2],
        'num_leaves': int(gbest_position[3]),
        'min_child_samples': int(gbest_position[4])
    }




# In[34]:


# Running MOPSO
best_hyperparams = mopso(X_train_selected, X_test_selected, y_train, y_test)


# In[ ]:


get_ipython().run_cell_magic('time', '', '# LightGBM model performance\nfrom sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Train the LightGBM algorithm with the best hyperparameters\nlg = lgb.LGBMClassifier(**best_hyperparams, random_state=42)\n\nt1 = time.time()\nlg.fit(X_train_selected, y_train)\nt2 = time.time()\nprint("Training time: ", t2-t1, "s")\n\nt3 = time.time()\ny_pred = lg.predict(X_test_selected)\nt4 = time.time()\nprint("Prediction time per sample: ", (t4-t3)/len(X_test)*1000, "ms")\n\nprint(classification_report(y_test,y_pred))\nprint("Accuracy of LightGBM: "+ str(accuracy_score(y_test, y_pred)))\nprint("Precision of LightGBM: "+ str(precision_score(y_test, y_pred, average=\'weighted\')))\nprint("Recall of LightGBM: "+ str(recall_score(y_test, y_pred, average=\'weighted\')))\nprint("Average F1 of LightGBM: "+ str(f1_score(y_test, y_pred, average=\'weighted\')))\nprint("F1 of LightGBM for each type of attack: "+ str(f1_score(y_test, y_pred, average=None)))\n\n# Plot the confusion matrix\ncm = confusion_matrix(y_test, y_pred)\nf, ax = plt.subplots(figsize=(5, 5))\nsns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax)\nplt.xlabel("y_pred")\nplt.ylabel("y_true")\nplt.show()\n')


# In[ ]:


# Save the LightGBM model and report its size and average confidence
import joblib
import os

joblib.dump(lg, "lightgbm_model_all.pkl")
lgb_model_size = os.path.getsize("lightgbm_model_all.pkl") / (1024 * 1024)
print(f"LightGBM Model Size: {lgb_model_size:.2f} MB")

# Predict probabilities
y_proba = lg.predict_proba(X_test_selected)

# Compute average confidence
avg_conf = np.mean(np.max(y_proba, axis=1))
print(f"Average Confidence: {avg_conf:.4f}")

from sklearn.calibration import calibration_curve
import numpy as np

def compute_ece(y_true, y_proba, n_bins=10):
    y_pred_class = np.argmax(y_proba, axis=1)
    correct = (np.array(y_true) == y_pred_class).astype(int)
    confidence = np.max(y_proba, axis=1)

    prob_true, prob_pred = calibration_curve(correct, confidence, n_bins=n_bins)
    bin_counts, _ = np.histogram(confidence, bins=n_bins)
    bin_weights = bin_counts[:len(prob_true)] / np.sum(bin_counts)

    ece = np.sum(np.abs(prob_true - prob_pred) * bin_weights)
    return ece

ece = compute_ece(y_test, y_proba)
print(f"ECE: {ece:.4f}")

from sklearn.utils import resample

def bootstrap_ci(data, n_bootstraps=1000, alpha=0.05):
    boot_means = [np.mean(resample(data)) for _ in range(n_bootstraps)]
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, upper

confidences = np.max(y_proba, axis=1)
ci_lower, ci_upper = bootstrap_ci(confidences)
print(f"95% CI for Avg Confidence: [{ci_lower:.4f}, {ci_upper:.4f}]")


# ### Optimize the XGBoost model

# In[38]:


# Display the default hyperparameters of XGBoost
xg.get_params()


# In[ ]:


# Write the MOPSO algorithm to optimize XGBoost hyperparameters

counter = 0  # Declare this at the top-level script

# Define the objective function (Optimizing F1 Score, Average Confidence, and Training Time)
def objective_function(position, X_train, X_test, y_train, y_test, f1_weight=0.95, confidence_weight=0.01, time_weight=0.04):
# def objective_function(position, X_train, X_test, y_train, y_test, f1_weight=0.8, confidence_weight=0.05, time_weight=0.15):
    global counter  # Declare counter as global to modify it
    counter += 1  # Increment counter

    start_time = time.time()

    # Apply hard constraints
    position[0] = np.clip(position[0], 50, 200)  # n_estimators
    position[1] = np.clip(position[1], 5, 50)    # max_depth
    position[2] = np.clip(position[2], 0.01, 0.3)  # learning_rate

    hyperparams = {
        'n_estimators': int(position[0]),
        'max_depth': int(position[1]),
        'learning_rate': position[2]
    }

    print(f"Evaluation {counter}: Hyperparameters before training:", hyperparams)  # Debugging line

    clf = xgb.XGBClassifier(**hyperparams, random_state=42, objective="multi:softprob", eval_metric="mlogloss")
    # clf.fit(X_train, y_train)
    clf.fit(X_train, y_train,         
    eval_set=[(X_test, y_test)],
        early_stopping_rounds=10, verbose=False)  # check whether to set early_stopping_rounds to reduce time

    predictions = clf.predict(X_test)
    proba = clf.predict_proba(X_test)

    f1 = f1_score(y_test, predictions, average='weighted')
    negative_f1 = -f1 * f1_weight

    avg_confidence = np.mean(np.max(proba, axis=1))
    negative_avg_confidence = -avg_confidence * confidence_weight

    elapsed_time = time.time() - start_time
    weighted_elapsed_time = elapsed_time * time_weight

    return negative_f1, negative_avg_confidence, weighted_elapsed_time

def initialize_swarm(num_particles):
    return [{'position': np.array([100, 10, 0.1]) + np.random.uniform(-10, 10, 3), 
             'velocity': np.random.uniform(-0.1, 0.1, 3)} for _ in range(num_particles)]

# Update velocity and position with adaptive inertia weight
def update_velocity_position(particle, pbest_position, gbest_position, iteration, max_iterations, 
                             w_max=0.9, w_min=0.4, c1=1.5, c2=1.5):

    # **Adaptive inertia weight**: Starts high, decreases over iterations for better convergence
    w = w_max - (w_max - w_min) * (iteration / max_iterations)

    inertia = w * particle['velocity']
    cognitive = c1 * np.random.random() * (pbest_position - particle['position'])
    social = c2 * np.random.random() * (gbest_position - particle['position'])

    # Compute new velocity
    new_velocity = inertia + cognitive + social

    # Compute new position
    new_position = particle['position'] + new_velocity

    # **Apply constraints to hyperparameters**
    new_position[0] = np.clip(new_position[0], 50, 200)  # n_estimators
    new_position[1] = np.clip(new_position[1], 5, 50)    # max_depth (Reduced upper limit)
    new_position[2] = np.clip(new_position[2], 0.01, 0.3)  # learning_rate (Narrowed range)

    # Update particle velocity and position
    particle['velocity'] = new_velocity
    particle['position'] = new_position



# Main MOPSO function
def mopso(X_train, X_test, y_train, y_test, num_particles=10, max_iterations=20):
    swarm = initialize_swarm(num_particles)
    gbest_score = np.array([float('inf')] * 3)
    gbest_position = None

    for iteration in range(max_iterations):
        for particle in swarm:
            fitness = np.array(objective_function(particle['position'], X_train, X_test, y_train, y_test))

            if np.sum(fitness) < np.sum(particle.get('pbest_score', np.array([float('inf')] * 3))):
                particle['pbest_score'] = fitness
                particle['pbest_position'] = particle['position']

            if np.sum(fitness) < np.sum(gbest_score):
                gbest_score = fitness
                gbest_position = particle['position']

        for particle in swarm:
            update_velocity_position(particle, particle['pbest_position'], gbest_position, iteration, max_iterations)

    print(f"Optimal hyperparameters are n_estimators: {int(gbest_position[0])}, max_depth: {int(gbest_position[1])}, learning_rate: {gbest_position[2]}")

    return {
        'n_estimators': int(gbest_position[0]),
        'max_depth': int(gbest_position[1]),
        'learning_rate': gbest_position[2]
    }




# In[41]:


# Running MOPSO
best_hyperparams = mopso(X_train_selected, X_test_selected, y_train, y_test)


# In[43]:


get_ipython().run_cell_magic('time', '', '# XGBoost model performance\nxg = xgb.XGBClassifier(**best_hyperparams, random_state=42, objective="multi:softprob", eval_metric="mlogloss")\n\n# X_train_x = X_train.values\n# X_test_x = X_test.values\n\nt1 = time.time()\nxg.fit(X_train_selected, y_train)\nt2 = time.time()\nprint("Training time: ", t2-t1, "s")\n\nt3 = time.time()\ny_pred = xg.predict(X_test_selected)\nt4 = time.time()\nprint("Prediction time per sample: ", (t4-t3)/len(X_test)*1000, "ms")\n\nprint(classification_report(y_test,y_pred))\nprint("Accuracy of XGBoost: "+ str(accuracy_score(y_test, y_pred)))\nprint("Precision of XGBoost: "+ str(precision_score(y_test, y_pred, average=\'weighted\')))\nprint("Recall of XGBoost: "+ str(recall_score(y_test, y_pred, average=\'weighted\')))\nprint("Average F1 of XGBoost: "+ str(f1_score(y_test, y_pred, average=\'weighted\')))\nprint("F1 of XGBoost for each type of attack: "+ str(f1_score(y_test, y_pred, average=None)))\nxg_f1=f1_score(y_test, y_pred, average=None)\n\n# Plot the confusion matrix\ncm=confusion_matrix(y_test,y_pred)\nf,ax=plt.subplots(figsize=(5,5))\nsns.heatmap(cm,annot=True,linewidth=0.5,linecolor="red",fmt=".0f",ax=ax)\nplt.xlabel("y_pred")\nplt.ylabel("y_true")\nplt.show()\n')


# In[44]:


# Save the XGBoost model and report its size and average confidence

# Save model
joblib.dump(xg, "xgboost_model_all.pkl")

# Check file size in MB
xgb_model_size = os.path.getsize("xgboost_model_all.pkl") / (1024 * 1024)
print(f"XGBoost Model Size: {xgb_model_size:.2f} MB")

# Get predicted probabilities
y_proba = xg.predict_proba(X_test_selected)

# Compute average confidence
avg_conf = np.mean(np.max(y_proba, axis=1))
print(f"Average Confidence: {avg_conf:.4f}")

from sklearn.calibration import calibration_curve
import numpy as np

def compute_ece(y_true, y_proba, n_bins=10):
    y_pred_class = np.argmax(y_proba, axis=1)
    correct = (np.array(y_true) == y_pred_class).astype(int)
    confidence = np.max(y_proba, axis=1)

    prob_true, prob_pred = calibration_curve(correct, confidence, n_bins=n_bins)
    bin_counts, _ = np.histogram(confidence, bins=n_bins)
    bin_weights = bin_counts[:len(prob_true)] / np.sum(bin_counts)

    ece = np.sum(np.abs(prob_true - prob_pred) * bin_weights)
    return ece

ece = compute_ece(y_test, y_proba)
print(f"ECE: {ece:.4f}")

from sklearn.utils import resample

def bootstrap_ci(data, n_bootstraps=1000, alpha=0.05):
    boot_means = [np.mean(resample(data)) for _ in range(n_bootstraps)]
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, upper

confidences = np.max(y_proba, axis=1)
ci_lower, ci_upper = bootstrap_ci(confidences)
print(f"95% CI for Avg Confidence: [{ci_lower:.4f}, {ci_upper:.4f}]")


