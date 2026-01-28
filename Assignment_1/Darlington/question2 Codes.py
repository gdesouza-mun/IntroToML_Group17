import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- 1. Load & Prepare Data (Same as your Q1 code) ---
# Assuming files are in current directory
train_sNC_features = pd.read_csv('train.sNC.csv', header=None)
train_sDAT_features = pd.read_csv('train.sDAT.csv', header=None)
test_sNC_features = pd.read_csv('test.sNC.csv', header=None)
test_sDAT_features = pd.read_csv('test.sDAT.csv', header=None)
grid_points = pd.read_csv('2D_grid_points.csv', header=None)

# Create Labels
y_train_sNC = pd.Series([0] * len(train_sNC_features))
y_train_sDAT = pd.Series([1] * len(train_sDAT_features))
y_test_sNC = pd.Series([0] * len(test_sNC_features))
y_test_sDAT = pd.Series([1] * len(test_sDAT_features))

# Concatenate
X_train = pd.concat([train_sNC_features, train_sDAT_features], axis=0, ignore_index=True)
y_train = pd.concat([y_train_sNC, y_train_sDAT], axis=0, ignore_index=True)
X_test = pd.concat([test_sNC_features, test_sDAT_features], axis=0, ignore_index=True)
y_test = pd.concat([y_test_sNC, y_test_sDAT], axis=0, ignore_index=True)

# Handle NaNs (Your Fix)
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())
grid_points = grid_points.fillna(grid_points.mean())

# --- 2. Step A: Find the Best k from Question 1 (Euclidean) ---
k_values = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]
best_k = None
min_test_error = float('inf')

print("Re-evaluating Q1 to find best k...")

for k in k_values:
    # Train Euclidean
    clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    clf.fit(X_train, y_train)

    # Predict
    y_test_pred = clf.predict(X_test)
    test_error = 1 - accuracy_score(y_test, y_test_pred)

    # Check if this is the best k
    if test_error < min_test_error:
        min_test_error = test_error
        best_k = k

print(f"Best k found: {best_k} (Test Error: {min_test_error:.4f})")

# --- 3. Step B: Run Q2 (Manhattan Distance) ---
# Train NEW classifier using best_k and metric='manhattan'
clf_manhattan = KNeighborsClassifier(n_neighbors=best_k, metric='manhattan')
clf_manhattan.fit(X_train, y_train)

# Predictions
y_train_pred_m = clf_manhattan.predict(X_train)
y_test_pred_m = clf_manhattan.predict(X_test)
grid_pred_m = clf_manhattan.predict(grid_points)

# Calculate Errors
manhattan_train_error = 1 - accuracy_score(y_train, y_train_pred_m)
manhattan_test_error = 1 - accuracy_score(y_test, y_test_pred_m)

# --- 4. Step C: Generate Visualization ---
# [cite: 66, 67]
colors = {0: 'green', 1: 'blue'}

plt.figure(figsize=(10, 8))

# 1. Decision Boundary (Grid)
plt.scatter(grid_points.iloc[:, 0], grid_points.iloc[:, 1],
            c=[colors[p] for p in grid_pred_m], marker='.', alpha=0.1, label='Boundary')

# 2. Training Data
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1],
            c=[colors[y] for y in y_train], marker='o', edgecolor='k', label='Train Data')

# 3. Test Data
plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1],
            c=[colors[y] for y in y_test], marker='+', s=100, linewidth=2, label='Test Data')

plt.xlabel("x1 (Isthmus Cingulate)")
plt.ylabel("x2 (Precuneus)")
plt.title(
    f'Manhattan kNN (k={best_k})\nTrain Error: {manhattan_train_error:.4f}, Test Error: {manhattan_test_error:.4f}')
plt.legend()
plt.show()

# Print comparison for discussion
print("--- Comparison Results ---")
print(f"Euclidean (k={best_k}) Test Error: {min_test_error:.4f}")
print(f"Manhattan (k={best_k}) Test Error: {manhattan_test_error:.4f}")