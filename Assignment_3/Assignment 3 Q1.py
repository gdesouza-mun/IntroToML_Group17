import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from sklearn.impute import SimpleImputer  # <-- Added import for handling NaNs

# ==========================================
# Step 1: Data Loading & Preparation
# ==========================================
# Load training datasets (assuming files are in your current working directory)
train_snc = pd.read_csv('train.fdg_pet.sNC.csv', header=None)
train_sdat = pd.read_csv('train.fdg_pet.sDAT.csv', header=None)

# Load testing datasets
test_snc = pd.read_csv('test.fdg_pet.sNC.csv', header=None)
test_sdat = pd.read_csv('test.fdg_pet.sDAT.csv', header=None)

# Combine the data.
X_train = pd.concat([train_snc, train_sdat], ignore_index=True)
# Create target labels: 0 for sNC and 1 for sDAT
y_train = np.concatenate([np.zeros(len(train_snc)), np.ones(len(train_sdat))])

X_test = pd.concat([test_snc, test_sdat], ignore_index=True)
y_test = np.concatenate([np.zeros(len(test_snc)), np.ones(len(test_sdat))])
# ==========================================
# Step 1.5: Handle Missing Data (NaNs)
# ==========================================
# Initialize the imputer to replace NaNs with the mean of the column
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data and transform it
X_train = imputer.fit_transform(X_train)

# Transform the test data using the same imputer to prevent data leakage
X_test = imputer.transform(X_test)

# ==========================================
# Step 2 & 3: Cross-Validation & Retraining
# ==========================================
# Define a range of values for the regularization parameter 'C'
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Initialize the linear SVM classifier
svm_linear = SVC(kernel='linear', random_state=42)

# Set up the Grid Search with 5-fold cross-validation
# We use balanced_accuracy for scoring as it's the primary metric for Question 4
grid_search = GridSearchCV(svm_linear, param_grid, cv=5, scoring='balanced_accuracy')

# Fit the model to find the best 'C'.
# Note: GridSearchCV automatically refits the best model on the entire training dataset!
grid_search.fit(X_train, y_train)

best_C = grid_search.best_params_['C']
print(f"Optimal 'C' value found: {best_C}")
best_model = grid_search.best_estimator_

# ==========================================
# Step 4: Model Evaluation
# ==========================================
# Predict on the unseen test data
y_pred = best_model.predict(X_test)

# Calculate standard metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred) # Recall is equivalent to Sensitivity
balanced_acc = balanced_accuracy_score(y_test, y_pred)

# Calculate Specificity using the confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print("\n--- Performance Metrics on Test Set ---")
print(f"Accuracy:          {accuracy:.4f}")
print(f"Sensitivity:       {sensitivity:.4f}")
print(f"Specificity:       {specificity:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {sensitivity:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")

# ==========================================
# Step 5: Plotting Performance vs. C
# ==========================================
# Extract CV scores for plotting
mean_cv_scores = grid_search.cv_results_['mean_test_score']
c_values = param_grid['C']

plt.figure(figsize=(8, 5))
plt.plot(c_values, mean_cv_scores, marker='o', linestyle='-', color='teal')
plt.xscale('log') # A logarithmic scale makes it easier to visualize the wide range of C values
plt.xlabel('Regularization Parameter (C)')
plt.ylabel('Cross-Validation Balanced Accuracy')
plt.title('Linear SVM: CV Performance vs. C')
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.tight_layout()
plt.show()