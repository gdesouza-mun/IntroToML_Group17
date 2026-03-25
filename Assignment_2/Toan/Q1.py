import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Q1_Ordinary_Least_Squares import OLS_V, OLS_CV
from matplotlib.ticker import LogLocator


# Load  train data
X_train = pd.read_csv('Data2/train.csv', header=None).to_numpy()
y_train = X_train[1:, -1].reshape(-1, 1)  # Last column is the label
X_train = X_train[1:, :-1]  # All columns except the last one are features
X_train = np.c_[np.ones(X_train.shape[0]), X_train] # Add intercept term
X_train = np.array(X_train)
y_train = np.array(y_train)
Err_val_method, RSE_val_method, R_squared_val_method, theta_val_method= OLS_V(X_train, y_train)
   
Err_CV_method, RSE_CV_method, R_squared_CV_method, theta_CV_method = OLS_CV(X_train, y_train, k=5)
print("Validation Method:")
print(f"Residual Standard Error: {RSE_val_method}")             
print(f"R-squared: {R_squared_val_method}")
print("\n5-Fold Cross-Validation Method:")
print(f"Average Residual Standard Error: {RSE_CV_method}")
print(f"Average R-squared: {R_squared_CV_method}")      

k_values = [ 2,3,4, 5, 6, 7,8,9,10]
RSE_list = []
R2_list = []

# Tính Err cho từng k
for k in k_values:
    Err_CV_method_k, RSE_CV_method_k, R_squared_CV_method_k, _ = OLS_CV(X_train, y_train, k=k)
    RSE_list.append(RSE_CV_method_k)
    R2_list.append(R_squared_CV_method_k)

# Vẽ đồ thị
fig, ax1 = plt.subplots(figsize=(8,5))

color = 'tab:blue'
ax1.set_xlabel('Number of folds (k)')
ax1.set_ylabel('RSE', color=color)
ax1.plot(k_values, RSE_list, marker='o', color=color, label='RSE- CV')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(k_values)

# Vẽ horizontal line cho RSE_val_method
ax1.axhline(y=RSE_val_method, color='tab:blue', linestyle='--', label=f'RSE - Validation ({RSE_val_method:.4f})')

# Tạo trục thứ 2 để vẽ R²
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('R-squared', color=color)
ax2.plot(k_values, R2_list, marker='s', color=color, label='R² - CV')
ax2.tick_params(axis='y', labelcolor=color)

# Vẽ horizontal line cho R_squared_val_method
ax2.axhline(y=R_squared_val_method, color='tab:red', linestyle='--', label=f'R² - Validation ({R_squared_val_method:.4f})')

# Thêm title và legend
fig.suptitle('RSE and R-squared vs Number of Folds (k)')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()