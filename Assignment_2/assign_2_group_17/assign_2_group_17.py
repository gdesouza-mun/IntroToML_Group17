#  Assignment 2
#
#  Group 17:
#  Darlington Nkrumah, MUN ID 202492437, dknkrumah@mun.ca
#  Greg de Souza, MUN ID 2025225,  gdesouza@mun.ca
#  Xuan Toan Doan, MUN ID 202583882, txdoan@mun.ca


####################################################################################
# Imports
####################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import math
import statistics

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

####################################################################################
# Helper Class
####################################################################################

class glb:
    feat_dic = [
        "Cement_component1__kgInAM_3Mixture_",
        "BlastFurnaceSlag_component2__kgInAM_3Mixture_",
        "FlyAsh_component3__kgInAM_3Mixture_",
        "Water_component4__kgInAM_3Mixture_",
        "Superplasticizer_component5__kgInAM_3Mixture_",
         "CoarseAggregate_component6__kgInAM_3Mixture_",
         "FineAggregate_component7__kgInAM_3Mixture_",
         "Age_day_"
    ]

    target_name = "ConcreteCompressiveStrength_MPa_Megapascals_"

    p_features=8

    train_Filepath="Data/train.csv"
    test_Filepath="Data/test.csv"

    alpha_space = [1.00000000e-04, 1.23284674e-04, 1.51991108e-04, 1.87381742e-04,
       2.31012970e-04, 2.84803587e-04, 3.51119173e-04, 4.32876128e-04,
       5.33669923e-04, 6.57933225e-04, 8.11130831e-04, 1.00000000e-03,
       1.23284674e-03, 1.51991108e-03, 1.87381742e-03, 2.31012970e-03,
       2.84803587e-03, 3.51119173e-03, 4.32876128e-03, 5.33669923e-03,
       6.57933225e-03, 8.11130831e-03, 1.00000000e-02, 1.23284674e-02,
       1.51991108e-02, 1.87381742e-02, 2.31012970e-02, 2.84803587e-02,
       3.51119173e-02, 4.32876128e-02, 5.33669923e-02, 6.57933225e-02,
       8.11130831e-02, 1.00000000e-01, 1.23284674e-01, 1.51991108e-01,
       1.87381742e-01, 2.31012970e-01, 2.84803587e-01, 3.51119173e-01,
       4.32876128e-01, 5.33669923e-01, 6.57933225e-01, 8.11130831e-01,
       1.00000000e+00, 1.23284674e+00, 1.51991108e+00, 1.87381742e+00,
       2.31012970e+00, 2.84803587e+00, 3.51119173e+00, 4.32876128e+00,
       5.33669923e+00, 6.57933225e+00, 8.11130831e+00, 1.00000000e+01,
       1.23284674e+01, 1.51991108e+01, 1.87381742e+01, 2.31012970e+01,
       2.84803587e+01, 3.51119173e+01, 4.32876128e+01, 5.33669923e+01,
       6.57933225e+01, 8.11130831e+01, 1.00000000e+02, 1.23284674e+02,
       1.51991108e+02, 1.87381742e+02, 2.31012970e+02, 2.84803587e+02,
       3.51119173e+02, 4.32876128e+02, 5.33669923e+02, 6.57933225e+02,
       8.11130831e+02, 1.00000000e+03, 1.23284674e+03, 1.51991108e+03,
       1.87381742e+03, 2.31012970e+03, 2.84803587e+03, 3.51119173e+03,
       4.32876128e+03, 5.33669923e+03, 6.57933225e+03, 8.11130831e+03,
       1.00000000e+04, 1.23284674e+04, 1.51991108e+04, 1.87381742e+04,
       2.31012970e+04, 2.84803587e+04, 3.51119173e+04, 4.32876128e+04,
       5.33669923e+04, 6.57933225e+04, 8.11130831e+04, 1.00000000e+05]


    def MSEtoRSE(MSE, sample_size, p=p_features):
        RSE = math.sqrt((sample_size*MSE)/(sample_size-p-1))

        return RSE


def calculate_rse(y_true, y_pred, p_features):
    n = len(y_true)
    rss = sum((y_true - y_pred) ** 2)
    return math.sqrt(rss / (n - p_features - 1))


####################################################################################
# Question 1
####################################################################################


def Q1_val(X,Y):
    #Generates r2 and RSE score for the validation approach, as well as the parameters for the model

    X_train, X_val, Y_train, Y_val = train_test_split(X,Y,
                                                      test_size=0.2,
                                                      random_state=10)

    lin_model= LinearRegression()
    lin_model.fit(X_train, Y_train)

    Y_val_pred = lin_model.predict(X_val)

    r2 = r2_score(Y_val, Y_val_pred)

    MSE = mean_squared_error(Y_val, Y_val_pred)
    n=len(Y_val)

    RSE = glb.MSEtoRSE(MSE, n)

    coefficients = lin_model.coef_
    intercept=lin_model.intercept_

    return r2, RSE, intercept, coefficients


def Q1_CV(X,Y):
    #Generates r2 and RSE for the CV Approach, with variance

    scoring = ["r2", "neg_mean_squared_error"]
    lin_model = LinearRegression()
    folds=5
    results = cross_validate(lin_model, X, Y, cv=folds, scoring=scoring)

    n=len(X)/folds

    r2_array = results['test_r2']
    MSE_array = -results["test_neg_mean_squared_error"]

    RSE_array = []
    for mse in MSE_array:
        RSE_array.append(glb.MSEtoRSE(mse, n))

    r2 = statistics.mean(r2_array)
    r2_var = statistics.variance(r2_array)
    RSE = statistics.mean(RSE_array)
    RSE_var = statistics.variance(RSE_array)

    return r2, RSE, r2_var, RSE_var


def Q1_results():

    #Load "train.csv" data
    df_train = pd.read_csv(glb.train_Filepath)

    #Separate target (Y) and features (X)
    Y=df_train[glb.target_name]
    X=df_train.drop(glb.target_name, axis=1)

    #Call respective assessment functions
    r2_val, RSE_val, inter_val, coef_val = Q1_val(X,Y)
    r2_CV, RSE_CV, r2_var, RSE_var = Q1_CV(X,Y)

    #Print Results
    print("    \t Split \t CV")
    print(f"R2 \t {r2_val:.4f}  {r2_CV:.4f} pm {r2_var:.4f}")
    print(f"RSE \t {RSE_val:.4f}  {RSE_CV:.4f} pm {RSE_var:.4f}")
    print(f"======== Best Parameters for OLS ======")
    print(f"Intercept \t {inter_val:.4}")
    for i in range(len(coef_val)):
        print(f"theta_{i} \t {coef_val[i]:.4f}")



def Q1_results():

    df_train = pd.read_csv('Data/train.csv')

    Y=df_train[glb.target_name]
    X=df_train.drop(glb.target_name, axis=1)

    r2_val, RSE_val, inter_val, coef_val = Q1_val(X,Y)
    r2_CV, RSE_CV, r2_var, RSE_var = Q1_CV(X,Y)

    print("    \t Split \t CV")
    print(f"R2 \t {r2_val:.4f}  {r2_CV:.4f} pm {r2_var:.4f}")
    print(f"RSE \t {RSE_val:.4f}  {RSE_CV:.4f} pm {RSE_var:.4f}")
    print(f"======== Best Parameters ======")
    print(f"Intercept \t {inter_val:.4}")
    for i in range(len(coef_val)):
        print(f"theta_{i} \t {coef_val[i]:.4f}")



####################################################################################
# Question 2
####################################################################################

def Q2_results():
    # 1. Load Data
    train_df = pd.read_csv(glb.train_Filepath)
    test_df = pd.read_csv(glb.test_Filepath)

    # Separate Features (X) and Target (y)
    X_train = train_df[glb.feat_dic]
    y_train = train_df[glb.target_name]
    X_test = test_df[glb.feat_dic]
    y_test = test_df[glb.target_name]


    # 2. Setup Ridge Regression with Grid Search
    # We define a range of alphas. Logspace is usually best for regularization parameters.
    # Checking alphas from 10^-4 to 10^4
    alphas = glb.alpha_space
    ridge = Ridge()

    # We use GridSearchCV to tune 'alpha'
    # cv=10 is a solid choice based on Question 1 discussion
    param_grid = {'alpha': alphas}
    folds=10
    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid,
                               scoring='neg_mean_squared_error', cv=10)
    grid_search.fit(X_train, y_train)

    # 3. Analyze Grid Search Results
    best_alpha = grid_search.best_params_['alpha']
    print(f"Best Alpha found: {best_alpha:.4f}")

    # Extract scores for plotting
    # GridSearchCV returns negative MSE, so we flip the sign to get positive MSE
    cv_scores_mse = -grid_search.cv_results_['mean_test_score']
    rse_scores = []

    for mse in cv_scores_mse:
        set_size=len(y_train)/folds
        rse_scores.append(glb.MSEtoRSE(mse, set_size))

    # 4. Retrain Final Ridge Model with Best Alpha
    final_ridge = Ridge(alpha=best_alpha)
    final_ridge.fit(X_train, y_train)

    # Predict on Test Set
    y_pred_ridge = final_ridge.predict(X_test)

    #5. Calculate Errors
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

    coef = final_ridge.coef_
    inter = final_ridge.intercept_

    print(f"======== Best Parameters for Ridge ======")
    print(f"Intercept \t {inter:.4}")
    for i in range(len(coef)):
        print(f"theta_{i} \t {coef[i]:.4f}")

    # 8. Plot Performance vs Alpha
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, rse_scores, marker='.', linestyle='-')
    plt.xlabel('Alpha (Regularization Parameter)')
    plt.ylabel('Relative Standard Error RSE (CV)')
    plt.title('Ridge Regression: Hyperparameter Tuning (Alpha vs RSE)')
    plt.axvline(best_alpha, color='r', linestyle='--', label=f'Best Alpha: {best_alpha:.2f}')
    plt.legend()
    plt.grid(True)
    print("Figure saved to Q2_results.png")
    plt.savefig("Q2_results.png", dpi=300)
    #plt.show()
    plt.close("All")

####################################################################################
# Question 3
####################################################################################

def Q3_results():
    # 1. Load Data
    train_df = pd.read_csv(glb.train_Filepath)
    test_df = pd.read_csv(glb.test_Filepath)

    # Separate Features (X) and Target (y)
    X_train = train_df[glb.feat_dic]
    y_train = train_df[glb.target_name]
    X_test = test_df[glb.feat_dic]
    y_test = test_df[glb.target_name]

    folds = 5
    # ---------------------------------------------------------
    # Part A: Run Previous Models (OLS & Ridge) for Comparison
    # ---------------------------------------------------------
    # OLS
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    y_pred_ols = ols.predict(X_test)
    r2_ols = r2_score(y_test, y_pred_ols)
    rse_ols = calculate_rse(y_test, y_pred_ols, glb.p_features)

    # Ridge (Using the alpha from Q2)
    best_ridge = Ridge(alpha=7742.6368)
    best_ridge.fit(X_train, y_train)
    y_pred_ridge = best_ridge.predict(X_test)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    rse_ridge = calculate_rse(y_test, y_pred_ridge, glb.p_features)

    # ---------------------------------------------------------
    # Part B: Lasso Regression Experiment (Question 3)
    # ---------------------------------------------------------

    # 1. Setup Lasso Grid Search
    alphas_lasso = glb.alpha_space
    lasso = Lasso(max_iter=10000, random_state=10) # Increased max_iter for convergence

    param_grid = {'alpha': alphas_lasso}
    lasso_search = GridSearchCV(estimator=lasso, param_grid=param_grid,
                                scoring='neg_mean_squared_error', cv=10)
    lasso_search.fit(X_train, y_train)

    # 2. Extract Results
    best_alpha_lasso = lasso_search.best_params_['alpha']
    cv_scores_mse_lasso = -lasso_search.cv_results_['mean_test_score']
    rse_scores = []

    for mse in cv_scores_mse_lasso:
        set_size=len(y_train)/folds
        rse_scores.append(glb.MSEtoRSE(mse, set_size))

    print(f"Best Lasso Alpha found: {best_alpha_lasso:.4f}")

    # 3. Retrain Final Lasso Model
    final_lasso = Lasso(alpha=best_alpha_lasso, max_iter=10000, random_state=42)
    final_lasso.fit(X_train, y_train)

    # Predict
    y_pred_lasso = final_lasso.predict(X_test)

    r2_lasso = r2_score(y_test, y_pred_lasso)
    rse_lasso = calculate_rse(y_test, y_pred_lasso, X_test.shape[1])

    # Check for Feature Selection (Zero coefficients)
    coef = final_lasso.coef_
    inter = final_lasso.intercept_

    print(f"\n--- Lasso Performance ---")
    print(f"R2 Score: {r2_lasso:.4f}")
    print(f"RSE: {rse_lasso:.4f}")
    print(f"======== Best Parameters for Lasso ======")
    print(f"Intercept \t {inter:.4}")
    for i in range(len(coef)):
        print(f"theta_{i} \t {coef[i]:.4f}")


    # ---------------------------------------------------------
    # Part C: Comparative Analysis
    # ---------------------------------------------------------
    print("\n--- Final Model Comparison (R2 Score) ---")
    print(f"OLS (Simple): {r2_ols:.4f}")
    print(f"Ridge:        {r2_ridge:.4f}")
    print(f"Lasso:        {r2_lasso:.4f}")

    print("\n--- Final Model Comparison (RSE Score) ---")
    print(f"OLS (Simple): {rse_ols:.4f}")
    print(f"Ridge:        {rse_ridge:.4f}")
    print(f"Lasso:        {rse_lasso:.4f}")


    # ---------------------------------------------------------
    # Part D: Plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas_lasso, rse_scores, marker='.', linestyle='-', color='green')
    plt.xlabel('Alpha (Regularization Parameter)')
    plt.ylabel('Relative Standard Error RSE (CV)')
    plt.title('Lasso Regression: Hyperparameter Tuning')
    plt.axvline(best_alpha_lasso, color='r', linestyle='--', label=f'Best Alpha: {best_alpha_lasso:.4f}')
    plt.legend()
    plt.grid(True)
    print("Figure saved to Q3_results.png")
    plt.savefig("Q3_results.png", dpi=300)
    #plt.show()
    plt.close("All")



####################################################################################
# Question 4
####################################################################################
def transform(X):
    #I'm transforming the data to have non-linear relation to y, based on some exploration
    X_new=X
    col0 = X_new[glb.feat_dic[0]].astype(float)
    col2 = X_new[glb.feat_dic[0]].astype(float)
    Age = X_new[glb.feat_dic[7]].astype(float)
    col4 = X_new[glb.feat_dic[4]].astype(float)
    col5 = X_new[glb.feat_dic[5]].astype(float)
    col6 = X_new[glb.feat_dic[6]].astype(float)
    col1_2 = X_new[glb.feat_dic[1]].astype(float)
    col_456 = []
    col_log0 = []
    col_2sqrt = []
    drop_index=[0,1,4,5,6,7]
    for i in drop_index:
        X_new=X_new.drop(glb.feat_dic[i], axis=1)


    for i in range(len(Age)):
        x0=col0[i]
        x2=col2[i]
        x4 = col4[i]
        x5 = col5[i]
        x6 = col6[i]
        x7=Age[i]
        Age[i] = math.log(x7)
        col_456.append(math.pow(x4*x5*x6, 0.5))
        col4[i] = math.pow(x2,1.66)
        col1_2[i] = math.pow(col1_2[i], 2)
        col_log0.append(math.log(x0))

    X_new["log(Age_day_)"] = Age
    X_new["x4**1.66"] = col4
    X_new["x1**2"] = col1_2
    X_new["x4*x5*x6"] = col_456
    X_new["logx0"] = col_log0

    return X_new

def predictCompressiveStrength(Xtest, data_dir):

    #Renaming the columns to avoid conflict in case Xtest is a pure array
    Xdf=pd.DataFrame(Xtest)
    Xdf.columns = glb.feat_dic

    path=data_dir
    if not path.endswith('/'):
        path+='/'

    df1 = pd.read_csv(path+'train.csv')
    df2 = pd.read_csv(path+'test.csv')
    df_train = pd.concat([df1, df2], axis=0, ignore_index=True)

    Y=df_train[glb.target_name]
    X=df_train.drop(glb.target_name, axis=1)

    X_new = transform(X)

    model = HistGradientBoostingRegressor(
        learning_rate=0.1,
        max_iter=500,  # Sufficient trees to converge
        max_depth=None,  # Allow trees to grow (regularized by min_samples_leaf)
        min_samples_leaf=20,  # Prevents overfitting to noise
        l2_regularization=0.1,  # Slight regularization
        random_state=42
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    pipeline.fit(X_new, Y)
    Xtest_new = transform(Xdf)
    Y_out = pipeline.predict(Xtest_new)

    return Y_out


#########################################################################################
# Calls to generate the results
#########################################################################################


if __name__=="__main__":
    print("Q1 Results:")
    Q1_results()
    print("Q2 Results:")
    Q2_results()
    print("Q3 Results:")
    Q3_results()
    print("Trying for Question 4")
    try:
        ytest = predictCompressiveStrength(Xtest, data_dir)

    except:
        print("Exception: predictCompressiveStrength arguments not well defined")
