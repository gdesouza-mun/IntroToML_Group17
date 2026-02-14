import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data
# Ensure 'train.csv' and 'test.csv' are in your directory
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Separate Features (X) and Target (y)
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# ---------------------------------------------------------
# Part A: Run Previous Models (OLS & Ridge) for Comparison
# ---------------------------------------------------------
# OLS
ols = LinearRegression()
ols.fit(X_train, y_train)
y_pred_ols = ols.predict(X_test)
r2_ols = r2_score(y_test, y_pred_ols)

# Ridge (Using the logic from Q2)
ridge_alphas = np.logspace(-4, 4, 100)
ridge_grid = GridSearchCV(Ridge(), {'alpha': ridge_alphas},
                          scoring='neg_mean_squared_error', cv=10)
ridge_grid.fit(X_train, y_train)
best_ridge = ridge_grid.best_estimator_
y_pred_ridge = best_ridge.predict(X_test)
r2_ridge = r2_score(y_test, y_pred_ridge)

# ---------------------------------------------------------
# Part B: Lasso Regression Experiment (Question 3)
# ---------------------------------------------------------

# 1. Setup Lasso Grid Search
# Lasso often needs a slightly different alpha range or max_iter adjustment
alphas_lasso = np.logspace(-4, 4, 100)
lasso = Lasso(max_iter=10000, random_state=42) # Increased max_iter for convergence

param_grid = {'alpha': alphas_lasso}
lasso_search = GridSearchCV(estimator=lasso, param_grid=param_grid,
                            scoring='neg_mean_squared_error', cv=10)
lasso_search.fit(X_train, y_train)

# 2. Extract Results
best_alpha_lasso = lasso_search.best_params_['alpha']
cv_scores_mse_lasso = -lasso_search.cv_results_['mean_test_score']

print(f"Best Lasso Alpha found: {best_alpha_lasso:.4f}")

# 3. Retrain Final Lasso Model
final_lasso = Lasso(alpha=best_alpha_lasso, max_iter=10000, random_state=42)
final_lasso.fit(X_train, y_train)

# Predict
y_pred_lasso = final_lasso.predict(X_test)

# 4. Calculate Metrics
def calculate_rse(y_true, y_pred, p_features):
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    return np.sqrt(rss / (n - p_features - 1))

r2_lasso = r2_score(y_test, y_pred_lasso)
rse_lasso = calculate_rse(y_test, y_pred_lasso, X_test.shape[1])

# Check for Feature Selection (Zero coefficients)
coefs = final_lasso.coef_
n_zero_coefs = np.sum(coefs == 0)
feature_names = X_train.columns
dropped_features = feature_names[coefs == 0]

print(f"\n--- Lasso Performance ---")
print(f"R2 Score: {r2_lasso:.4f}")
print(f"RSE: {rse_lasso:.4f}")
print(f"Features reduced to zero: {n_zero_coefs}")
if n_zero_coefs > 0:
    print(f"Dropped Features: {list(dropped_features)}")

# ---------------------------------------------------------
# Part C: Comparative Analysis
# ---------------------------------------------------------
print("\n--- Final Model Comparison (R2 Score) ---")
print(f"OLS (Simple): {r2_ols:.4f}")
print(f"Ridge:        {r2_ridge:.4f}")
print(f"Lasso:        {r2_lasso:.4f}")

# ---------------------------------------------------------
# Part D: Plotting
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.semilogx(alphas_lasso, cv_scores_mse_lasso, marker='.', linestyle='-', color='green')
plt.xlabel('Alpha (Regularization Parameter)')
plt.ylabel('Mean Squared Error (CV)')
plt.title('Lasso Regression: Hyperparameter Tuning')
plt.axvline(best_alpha_lasso, color='r', linestyle='--', label=f'Best Alpha: {best_alpha_lasso:.4f}')
plt.legend()
plt.grid(True)
plt.show()