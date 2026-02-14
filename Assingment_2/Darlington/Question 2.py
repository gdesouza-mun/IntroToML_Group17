import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
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

# 2. Setup Ridge Regression with Grid Search
# We define a range of alphas. Logspace is usually best for regularization parameters.
# Checking alphas from 10^-4 to 10^4
alphas = np.logspace(-4, 4, 100)
ridge = Ridge()

# We use GridSearchCV to tune 'alpha'
# cv=10 is a solid choice based on Question 1 discussion
param_grid = {'alpha': alphas}
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=10)
grid_search.fit(X_train, y_train)

# 3. Analyze Grid Search Results
best_alpha = grid_search.best_params_['alpha']
print(f"Best Alpha found: {best_alpha:.4f}")

# Extract scores for plotting
# GridSearchCV returns negative MSE, so we flip the sign to get positive MSE
cv_scores_mse = -grid_search.cv_results_['mean_test_score']

# 4. Retrain Final Ridge Model with Best Alpha
final_ridge = Ridge(alpha=best_alpha)
final_ridge.fit(X_train, y_train)

# Predict on Test Set
y_pred_ridge = final_ridge.predict(X_test)

# 5. Calculate Ridge Performance Metrics
def calculate_rse(y_true, y_pred, p_features):
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    return np.sqrt(rss / (n - p_features - 1))

r2_ridge = r2_score(y_test, y_pred_ridge)
rse_ridge = calculate_rse(y_test, y_pred_ridge, X_test.shape[1])

# 6. Compare with Simple Linear Regression (OLS)
# We must train OLS on the full training set and test on 'test.csv' for a fair comparison
ols = LinearRegression()
ols.fit(X_train, y_train)
y_pred_ols = ols.predict(X_test)

r2_ols = r2_score(y_test, y_pred_ols)
rse_ols = calculate_rse(y_test, y_pred_ols, X_test.shape[1])

# 7. Output Results
print("\n--- Model Comparison on Test Set (test.csv) ---")
print(f"OLS (Simple) - R2: {r2_ols:.4f}, RSE: {rse_ols:.4f}")
print(f"Ridge (Best) - R2: {r2_ridge:.4f}, RSE: {rse_ridge:.4f}")
print(f"Difference (Ridge - OLS) - R2: {r2_ridge - r2_ols:.4f}")

# 8. Plot Performance vs Alpha
plt.figure(figsize=(10, 6))
plt.semilogx(alphas, cv_scores_mse, marker='.', linestyle='-')
plt.xlabel('Alpha (Regularization Parameter)')
plt.ylabel('Mean Squared Error (CV)')
plt.title('Ridge Regression: Hyperparameter Tuning (Alpha vs MSE)')
plt.axvline(best_alpha, color='r', linestyle='--', label=f'Best Alpha: {best_alpha:.2f}')
plt.legend()
plt.grid(True)
plt.show()