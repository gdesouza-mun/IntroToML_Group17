import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load Data
# Assuming train.csv is in the same directory
data_path = 'train.csv'
df = pd.read_csv(data_path)

# Separate Target (y) and Features (X)
# Note: Check your CSV column names. Assuming last column is target.
X = df.iloc[:, :-1]  # All columns except the last
y = df.iloc[:, -1]  # The last column (Compressive Strength)

# ==========================================
# PART A: Validation Approach (Hold-out)
# ==========================================
print("--- Validation Approach ---")

# The prompt asks to split into train/val/test and train on train+val.
# Practically, this is a single split where we hold out a 'test' set.
# Let's use 20% for the hold-out test set.
X_train_val, X_test_holdout, y_train_val, y_test_holdout = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Multivariate OLS Linear Regression Model
model = LinearRegression()
model.fit(X_train_val, y_train_val)

# Predict
y_pred_holdout = model.predict(X_test_holdout)

# Calculate Metrics
# R^2
r2_val = r2_score(y_test_holdout, y_pred_holdout)

# RSE Calculation
n = len(y_test_holdout)
p = X_train_val.shape[1]  # number of features (8)
rss = np.sum((y_test_holdout - y_pred_holdout) ** 2)
rse_val = np.sqrt(rss / (n - p - 1))

print(f"Validation Approach R^2: {r2_val:.4f}")
print(f"Validation Approach RSE: {rse_val:.4f}")

# ==========================================
# PART B: Cross-Validation (CV) Approach
# ==========================================
print("\n--- Cross-Validation Approach ---")

# Discussion point for report: Why 10 folds?
# 800 samples is sufficient for 10-fold (80 in validation).
# It provides a robust estimate with lower bias than 5-fold.
k_folds = 10
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

cv_r2_scores = []
cv_rse_scores = []

model_cv = LinearRegression()

# Iterate through folds manually to calculate RSE correctly per fold
for train_index, test_index in kf.split(X):
    X_fold_train, X_fold_test = X.iloc[train_index], X.iloc[test_index]
    y_fold_train, y_fold_test = y.iloc[train_index], y.iloc[test_index]

    # Train
    model_cv.fit(X_fold_train, y_fold_train)

    # Predict
    y_fold_pred = model_cv.predict(X_fold_test)

    # R^2
    cv_r2_scores.append(r2_score(y_fold_test, y_fold_pred))

    # RSE
    n_fold = len(y_fold_test)
    rss_fold = np.sum((y_fold_test - y_fold_pred) ** 2)
    rse_fold = np.sqrt(rss_fold / (n_fold - p - 1))
    cv_rse_scores.append(rse_fold)

print(f"CV Average R^2: {np.mean(cv_r2_scores):.4f}")
print(f"CV Average RSE: {np.mean(cv_rse_scores):.4f}")

# ==========================================
# Discussion Points for Report
# ==========================================
# 1. Compare 'r2_val' with 'np.mean(cv_r2_scores)'.
# 2. Compare 'rse_val' with 'np.mean(cv_rse_scores)'.
# 3. Discuss which estimate is more reliable (usually CV).