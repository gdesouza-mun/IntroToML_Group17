import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- 1. Load feature datasets ---
train_sNC_features = pd.read_csv('train.sNC.csv', header=None)
train_sDAT_features = pd.read_csv('train.sDAT.csv', header=None)
test_sNC_features = pd.read_csv('test.sNC.csv', header=None)
test_sDAT_features = pd.read_csv('test.sDAT.csv', header=None)
grid_points = pd.read_csv('2D_grid_points.csv', header=None)

# --- 2. Prepare Data & Labels ---
y_train_sNC = pd.Series([0] * len(train_sNC_features))
y_train_sDAT = pd.Series([1] * len(train_sDAT_features))
y_test_sNC = pd.Series([0] * len(test_sNC_features))
y_test_sDAT = pd.Series([1] * len(test_sDAT_features))

X_train = pd.concat([train_sNC_features, train_sDAT_features], axis=0, ignore_index=True)
y_train = pd.concat([y_train_sNC, y_train_sDAT], axis=0, ignore_index=True)
X_test = pd.concat([test_sNC_features, test_sDAT_features], axis=0, ignore_index=True)
y_test = pd.concat([y_test_sNC, y_test_sDAT], axis=0, ignore_index=True)

# --- 3. FIX: Handle NaNs (The Error Fix) ---
# We drop rows with NaNs and ensure the labels (y) stay aligned
train_mask = X_train.notna().all(axis=1)
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

test_mask = X_test.notna().all(axis=1)
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# Also ensure the grid points are clean
grid_points = grid_points.fillna(grid_points.mean())



# --- 4. Plotting Loop ---
colors = {0: 'green', 1: 'blue'}
k_values = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]

if X_train.empty:
    print("Error: X_train is empty! Check your CSV files for valid data.")
else:
    # ... (Keep your data loading and concatenation from the fix above) ...

    for k in k_values:
        if k > len(X_train):
            print(f"Skipping k={k}")
            continue

        # Initialize and Fit
        clf = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        clf.fit(X_train, y_train)

        # Predictions (Moved INSIDE the loop)
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        grid_pred = clf.predict(grid_points)

        # Calculate Error Rates
        train_error = 1 - accuracy_score(y_train, y_train_pred)
        test_error = 1 - accuracy_score(y_test, y_test_pred)

        # Visualization
        plt.figure(figsize=(10, 8))

        # 1. Decision Boundary (The background)
        plt.scatter(grid_points.iloc[:, 0], grid_points.iloc[:, 1],
                    c=[colors[p] for p in grid_pred], marker='.', alpha=0.1)

        # 2. Training Data
        plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1],
                    c=[colors[y] for y in y_train], marker='o', edgecolor='k', label='Train Data')

        # 3. Test Data
        plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1],
                    c=[colors[y] for y in y_test], marker='+', s=100, linewidth=2, label='Test Data')

        plt.title(f'kNN (k={k})\nTrain Error: {train_error:.4f}, Test Error: {test_error:.4f}')
        plt.legend()
        plt.show()