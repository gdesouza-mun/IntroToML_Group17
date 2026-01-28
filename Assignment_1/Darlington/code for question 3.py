import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- 1. Configuration ---
# Set this to 'euclidean' or 'manhattan' based on your Q1 vs Q2 results
chosen_metric = 'euclidean'

# --- 2. Load Data ---
train_sNC = pd.read_csv('train.sNC.csv', header=None)
train_sDAT = pd.read_csv('train.sDAT.csv', header=None)
test_sNC = pd.read_csv('test.sNC.csv', header=None)
test_sDAT = pd.read_csv('test.sDAT.csv', header=None)

# Create Labels and Concatenate
X_train = pd.concat([train_sNC, train_sDAT], axis=0, ignore_index=True)
y_train = pd.concat([pd.Series([0] * len(train_sNC)), pd.Series([1] * len(train_sDAT))], axis=0, ignore_index=True)

X_test = pd.concat([test_sNC, test_sDAT], axis=0, ignore_index=True)
y_test = pd.concat([pd.Series([0] * len(test_sNC)), pd.Series([1] * len(test_sDAT))], axis=0, ignore_index=True)

# Handle NaNs (using mean imputation as before)
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# --- 3. Explore Parameter Space ---
# We need 1/k to range from 0.01 to 1.00.
# This means k ranges from 100 (1/100=0.01) down to 1 (1/1=1.0).
k_values = range(1, 101)
model_capacity = []  # This will store 1/k
train_errors = []
test_errors = []

print(f"Generating plot using {chosen_metric} distance...")

for k in k_values:
    # Calculate Model Capacity (1/k)
    capacity = 1.0 / k
    model_capacity.append(capacity)

    # Train
    clf = KNeighborsClassifier(n_neighbors=k, metric=chosen_metric)
    clf.fit(X_train, y_train)

    # Predict & Calculate Error
    train_err = 1 - accuracy_score(y_train, clf.predict(X_train))
    test_err = 1 - accuracy_score(y_test, clf.predict(X_test))

    train_errors.append(train_err)
    test_errors.append(test_err)

# --- 4. Plotting "Error Rate vs Model Capacity" ---
plt.figure(figsize=(10, 6))

# Plot Curves
plt.plot(model_capacity, train_errors, label='Training Error', color='cyan', marker='.', linestyle='-')
plt.plot(model_capacity, test_errors, label='Test Error', color='orange', marker='.', linestyle='--')

# Formatting as per Lecture 4 Slide 3
plt.xscale('log')  # Log scale for x-axis
plt.xlabel(r'Model Capacity ($\frac{1}{k}$)', fontsize=12)
plt.ylabel('Error Rate', fontsize=12)
plt.title(f'Error Rate vs. Model Capacity ({chosen_metric})', fontsize=14)
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.3)

# Invert x-axis if necessary to match slide direction?
# The slide usually goes low->high capacity (left to right).
# 0.01 (Low Capacity) -> 1.00 (High Capacity).
# Matplotlib log scale does this naturally (0.01 on left, 1.0 on right).

plt.show()