import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from sklearn.impute import SimpleImputer

# ==========================================
# Step 1: Data Loading, Preparation & Imputation
# ==========================================
# Load datasets ensuring no header is assumed
train_snc = pd.read_csv('train.fdg_pet.sNC.csv', header=None)
train_sdat = pd.read_csv('train.fdg_pet.sDAT.csv', header=None)
test_snc = pd.read_csv('test.fdg_pet.sNC.csv', header=None)
test_sdat = pd.read_csv('test.fdg_pet.sDAT.csv', header=None)

# Combine the data
X_train = pd.concat([train_snc, train_sdat], ignore_index=True)
y_train = np.concatenate([np.zeros(len(train_snc)), np.ones(len(train_sdat))])

X_test = pd.concat([test_snc, test_sdat], ignore_index=True)
y_test = np.concatenate([np.zeros(len(test_snc)), np.ones(len(test_sdat))])

# Impute missing values (NaNs) with the column mean
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# ==========================================
# Step 2 & 3: Polynomial SVM Grid Search & Retraining
# ==========================================
# Define ranges for 'C' and degree 'd'
param_grid = {
    'C': [0.1, 1, 10, 100],
    'degree': [2, 3, 4] # Common polynomial degrees to test
}

# Initialize the Polynomial SVM classifier
svm_poly = SVC(kernel='poly', random_state=42)

# Set up the Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(svm_poly, param_grid, cv=5, scoring='balanced_accuracy')

print("Starting Grid Search for Polynomial Kernel... (This might take a minute)")
grid_search.fit(X_train, y_train)

best_C = grid_search.best_params_['C']
best_degree = grid_search.best_params_['degree']
print(f"\nOptimal 'C' value found: {best_C}")
print(f"Optimal 'degree' (d) found: {best_degree}")

best_model = grid_search.best_estimator_

# ==========================================
# Step 4: Model Evaluation
# ==========================================
y_pred = best_model.predict(X_test)

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
