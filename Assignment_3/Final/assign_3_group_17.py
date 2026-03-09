#  Assignment 3
#
#  Group 17:
#  Darlington Nkrumah, MUN ID 202492437, dknkrumah@mun.ca
#  Greg de Souza, MUN ID 2025225,  gdesouza@mun.ca
#  Xuan Toan Doan, MUN ID 202583882, txdoan@mun.ca


####################################################################################
# Imports
####################################################################################
import sys
import os
import math

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from sklearn.impute import SimpleImputer  # <-- Added import fr handling NaNs
from sklearn.preprocessing import StandardScaler

# Global Utilities

# Standard variables so we can change stuff globally
class Global:
    train_snc_path="Data/train.fdg_pet.sNC.csv"
    train_sdat_path="Data/train.fdg_pet.sDAT.csv"

    test_snc_path="Data/test.fdg_pet.sNC.csv"
    test_sdat_path="Data/test.fdg_pet.sDAT.csv"

    snc_label=0
    sdat_label=1

    feature_names=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
                   "x10", "x11", "x12", "x13", "x14"]

    random_seed=17
    cv_folds=10
    main_score='balanced_accuracy'

#We can't import numpy :(
def generate_logspace(start, stop, num=50, endpoint=True, base=10.0):
    #Gen Log space from exp(start) to exp(stop)
    if num <= 0:
        return []

    # 1. Create a linear space (like numpy.linspace) for the exponents
    if endpoint:
        # Include the stop value in the linear range
        step = (stop - start) / (num - 1) if num > 1 else 0
        linear_space = [start + step * i for i in range(num)]
    else:
        # Exclude the stop value
        step = (stop - start) / num
        linear_space = [start + step * i for i in range(num)]

    # 2. Convert the linear space of exponents back to the original base
    log_space = [base ** exp for exp in linear_space]

    return log_space

#So we access each model identically
def print_assessement(y_test, y_pred, model_name=""):
    # Calculate standard metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred) # Recall is equivalent to Sensitivity
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    # Calculate Specificity using the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    print(f"\n--- Performance Metrics on Test Set for {model_name}---")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Sensitivity:       {sensitivity:.4f}")
    print(f"Specificity:       {specificity:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {sensitivity:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

def load_data():

    train_snc = pd.read_csv(Global.train_snc_path, header=None)
    train_snc.columns=Global.feature_names
    train_snc["y"]=Global.snc_label

    train_sdat = pd.read_csv(Global.train_sdat_path, header=None)
    train_sdat.columns=Global.feature_names
    train_sdat["y"]=Global.sdat_label

    train_df=pd.concat([train_snc, train_sdat], ignore_index=True)

    test_snc = pd.read_csv(Global.test_snc_path, header=None)
    test_snc.columns=Global.feature_names
    test_snc["y"]=Global.snc_label

    test_sdat = pd.read_csv(Global.test_sdat_path, header=None)
    test_sdat.columns=Global.feature_names
    test_sdat["y"]=Global.sdat_label

    test_df=pd.concat([test_snc, test_sdat], ignore_index=True)

    y_train=train_df["y"]
    X_train=train_df.drop(columns=["y"])

    y_test=test_df["y"]
    X_test=test_df.drop(columns=["y"])

    return X_train, y_train, X_test, y_test




####################################################################################
# Question 1
####################################################################################
def Q1_results():

    X_train, y_train, X_test, y_test = load_data()

    #print(X_test)

    #Getting a log space from 10^-3 to 10^3
    C_logspace = generate_logspace(-3, 3)
    param_grid={'C':C_logspace}

    svm_linear = SVC(kernel='linear', random_state=Global.random_seed)
    grid_search = GridSearchCV(svm_linear, param_grid, cv=Global.cv_folds,
                               scoring=Global.main_score, refit=True)

    grid_search.fit(X_train, y_train)
    best_c=grid_search.best_params_['C']
    print(f"Best C={best_c:.4f}")
    best_model = grid_search.best_estimator_
    y_pred=best_model.predict(X_test)

    print_assessement(y_test, y_pred, "Linear SVM")

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
    # plt.savefig("Q1_results.png", dpi=300)
    # print("Plot Saved to Q1_results.png")
    plt.show()


####################################################################################
# Question 2
####################################################################################
def Q2_results():
    X_train, y_train, X_test, y_test = load_data()

    C_logspace = generate_logspace(-3, -1 , 20)
    #d_values = [2,3,4,5]
    d_values = [4]
    param_grid = {
        'C': C_logspace,
        'degree': d_values
        }

    svm_poly=SVC(kernel='poly', random_state=Global.random_seed)
    grid_search=GridSearchCV(svm_poly, param_grid,
                             cv=Global.cv_folds,scoring=Global.main_score,
                             verbose=1, n_jobs=-1)

    print("Starting Grid Search for Polynomial Kernel, this might take a minute")
    print("We explored other values for C and d, but here we'll make a shorter search to save you time")
    grid_search.fit(X_train, y_train)

    best_C = grid_search.best_params_['C']
    best_degree = grid_search.best_params_['degree']
    print(f"\nOptimal 'C' value found: {best_C}")
    print(f"Optimal 'degree' (d) found: {best_degree}")

    best_model=grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print_assessement(y_test, y_pred, "Polynomial Kernel SVM")



####################################################################################
# Question 3
####################################################################################
def Q3_results():
    X_train, y_train, X_test, y_test = load_data()

    C_logspace = generate_logspace(-2, 2 , 20)
    gamma_logspace=generate_logspace(-2, 2, 20)
    param_grid = {
        'C': C_logspace,
        'gamma': gamma_logspace
        }

    # Initialize the RBF SVM classifier
    svm_rbf = SVC(kernel='rbf', random_state=42)

    # Set up the Grid Search with 5-fold cross-validation
    grid_search = GridSearchCV(svm_rbf, param_grid,
                               cv=Global.cv_folds, scoring=Global.main_score,
                               verbose=1, n_jobs=-1)

    print("Starting Grid Search for RBF Kernel... (This will also take a moment)")
    grid_search.fit(X_train, y_train)

    best_C = grid_search.best_params_['C']
    best_gamma = grid_search.best_params_['gamma']
    print(f"\nOptimal 'C' value found: {best_C}")
    print(f"Optimal 'gamma' found: {best_gamma}")

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    print_assessement(y_test, y_pred, "RBF Kernel SVM")


####################################################################################
# Question 4
####################################################################################
def ytest_diagnoseDAT(Xtest, data_dir):
    """
    Returns a vector of predictions with elements "0" for sNC and "1" for sDAT,
    corresponding to each of the N_test features vectors in Xtest.
    """

    train_snc_path = os.path.join(data_dir, 'train.fdg_pet.sNC.csv')
    train_sdat_path = os.path.join(data_dir, 'train.fdg_pet.sDAT.csv')
    test_snc_path = os.path.join(data_dir, 'test.fdg_pet.sNC.csv')
    test_sdat_path = os.path.join(data_dir, 'test.fdg_pet.sDAT.csv')

    train_snc = pd.read_csv(train_snc_path, header=None)
    train_snc.columns=Global.feature_names
    train_snc["y"]=Global.snc_label

    train_sdat = pd.read_csv(train_sdat_path, header=None)
    train_sdat.columns=Global.feature_names
    train_sdat["y"]=Global.sdat_label

    train_df=pd.concat([train_snc, train_sdat], ignore_index=True)

    test_snc = pd.read_csv(test_snc_path, header=None)
    test_snc.columns=Global.feature_names
    test_snc["y"]=Global.snc_label

    test_sdat = pd.read_csv(test_sdat_path, header=None)
    test_sdat.columns=Global.feature_names
    test_sdat["y"]=Global.sdat_label

    test_df=pd.concat([test_snc, test_sdat], ignore_index=True)

    df_all_train=pd.concat([train_df, test_df], ignore_index=True)

    y_all_train=df_all_train["y"]
    X_all_train = df_all_train.drop(columns=["y"])

    # 4. Handle Missing Data (NaNs)
    imputer = SimpleImputer(strategy='mean')
    X_all_train_imputed = imputer.fit_transform(X_all_train)

    # Ensure the Xtest passed into the function is also imputed
    Xtest_imputed = imputer.transform(Xtest)

    # 5. Scale the features (Crucial for the Polynomial kernel's performance)
    scaler = StandardScaler()
    X_all_train_scaled = scaler.fit_transform(X_all_train_imputed)

    # Scale the Xtest data using the same scaler
    Xtest_scaled = scaler.transform(Xtest_imputed)

    # 6. Initialize your BEST model (Polynomial: C=10, degree=3)
    best_model = SVC(kernel='poly', C=0.0033598, degree=4, random_state=17)

    # 7. Train the model on the fully combined, imputed, and scaled dataset
    best_model.fit(X_all_train_scaled, y_all_train)

    # 8. Predict on the provided Xtest
    predictions = best_model.predict(Xtest_scaled)

    return predictions


#########################################################################################
# Calls to generate the results
#########################################################################################
if __name__=="__main__":
    Q1_results()
    Q2_results()
    Q3_results()

    try:
        print("Starting diagnoseDat(Xtest, data_dir)")
        ytest=diagnoseDAT(Xtest, data_dir)
    except:
        print("Exception: diagnoseDat arguments not well defined")
