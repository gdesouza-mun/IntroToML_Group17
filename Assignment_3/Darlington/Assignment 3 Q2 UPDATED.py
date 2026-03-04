import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler # <-- New import for scaling

# ==========================================
# Step 1: Data Loading & Preparation
# ==========================================
train_snc = pd.read_csv('train.fdg_pet.sNC.csv', header=None)
train_sdat = pd.read_csv('train.fdg_pet.sDAT.csv', header=None)
test_snc = pd.read_csv('test.fdg_pet.sNC.csv', header=None)
test_sdat = pd.read_csv('test.fdg_pet.sDAT.csv', header=None)

X_train = pd.concat([train_snc, train_sdat], ignore_index=True)
y_train = np.concatenate([np.zeros(len(train_snc)), np.ones(len(train_sdat))])

X_test = pd.concat([test_snc, test_sdat], ignore_index=True)
y_test = np.concatenate([np.zeros(len(test_snc)), np.ones(len(test_sdat))])

# ==========================================
# Step 1.5: Imputation AND Scaling
# ==========================================
# 1. Handle missing values first
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# 2. Scale the features to speed up polynomial convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# ==========================================
# Step 2 & 3: Polynomial SVM Grid Search
# ==========================================
param_grid = {
    'C': [0.1, 1, 10, 100],
    'degree': [2, 3, 4]
}

svm_poly = SVC(kernel='poly', random_state=42)
grid_search = GridSearchCV(svm_poly, param_grid, cv=5, scoring='balanced_accuracy')

print("Starting Grid Search for Polynomial Kernel (with scaled features)...")
# Note that we are now fitting on X_train_scaled
grid_search.fit(X_train_scaled, y_train)

best_C = grid_search.best_params_['C']
best_degree = grid_search.best_params_['degree']
print(f"\nOptimal 'C' value found: {best_C}")
print(f"Optimal 'degree' (d) found: {best_degree}")

best_model = grid_search.best_estimator_

# ==========================================
# Step 4: Model Evaluation
# ==========================================
# Note that we must predict using X_test_scaled
y_pred = best_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print("\n--- Performance Metrics on Test Set (Polynomial Kernel) ---")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Sensitivity:       {sensitivity:.4f}")
print(f"Specificity:       {specificity:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {sensitivity:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")