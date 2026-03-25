import pandas as pd
import matplotlib.pyplot as plt
from dataVisu import glb
import math

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LassoLars
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Regularize the Data for sure
# Visualize Regularized Data
# Check Review Article?
# Check Scikit Linear Models

def regularized_net(X, Y, grid):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('en', ElasticNet(fit_intercept=True))
    ])

    folds=10
    param_grid = {'en__alpha': grid, 'en__l1_ratio':[0.05, 0.1, 0.2, 0.3]}
    grid = GridSearchCV(pipeline, param_grid, cv=folds,
                        scoring=["neg_mean_squared_error", "r2"],
                        refit='r2')
    grid.fit(X, Y)

    best_idx = grid.best_index_
    best_r2 = grid.cv_results_['mean_test_r2'][best_idx]
    best_mse = -grid.cv_results_['mean_test_neg_mean_squared_error'][best_idx]
    sample_size = len(X)/folds
    p = X.shape[1]
    #print(p)
    RSE = glb.MSEtoRSE(best_mse, sample_size, p)

    #print(grid.cv_results_)

    best_pipe = grid.best_estimator_
    best_net=best_pipe.named_steps['en']

    print(f"Best alpha = {grid.best_params_['en__alpha']}")
    print(f"Best l1 = {grid.best_params_['en__l1_ratio']}")
    print(f"Best r2 =  {best_r2:.4f}")
    coef = best_net.coef_
    inter = best_net.intercept_
    print(f"Best Intercept {inter:.4f}")
    for i in range(len(coef)):
        print(f"theta_{i} \t {coef[i]:.4f}")


def regularized_lasso(X, Y, grid):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(fit_intercept=True))
    ])

    folds=10
    param_grid = {'lasso__alpha': grid}
    grid = GridSearchCV(pipeline, param_grid, cv=folds,
                        scoring=["neg_mean_squared_error", "r2"],
                        refit='r2')
    grid.fit(X, Y)

    best_idx = grid.best_index_
    #print(grid.cv_results_)
    best_r2 = grid.cv_results_['mean_test_r2'][best_idx]
    best_mse = -grid.cv_results_['mean_test_neg_mean_squared_error'][best_idx]
    sample_size = len(X)/folds
    p = X.shape[1]
    #print(p)
    RSE = glb.MSEtoRSE(best_mse, sample_size, p)

    #print(grid.cv_results_)

    best_pipe = grid.best_estimator_
    best_lasso=best_pipe.named_steps['lasso']

    print(f"Best alpha = {grid.best_params_['lasso__alpha']}")
    print(f"Best r2 =  {best_r2:.4f}")
    coef = best_lasso.coef_
    inter = best_lasso.intercept_
    print(f"Best Intercept {inter:.4f}")
    for i in range(len(coef)):
        print(f"theta_{i} \t {coef[i]:.4f}")

def regularized_llar(X, Y, grid):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', LassoLars(fit_intercept=True))
    ])

    folds=10
    param_grid = {'lasso__alpha': grid}
    grid = GridSearchCV(pipeline, param_grid, cv=folds,
                        scoring=["neg_mean_squared_error", "r2"],
                        refit='r2')
    grid.fit(X, Y)

    best_idx = grid.best_index_
    #print(grid.cv_results_)
    best_r2 = grid.cv_results_['mean_test_r2'][best_idx]
    best_mse = -grid.cv_results_['mean_test_neg_mean_squared_error'][best_idx]
    sample_size = len(X)/folds
    p = X.shape[1]
    #print(p)
    RSE = glb.MSEtoRSE(best_mse, sample_size, p)

    #print(grid.cv_results_)

    best_pipe = grid.best_estimator_
    best_lasso=best_pipe.named_steps['lasso']

    print(f"Best alpha = {grid.best_params_['lasso__alpha']}")
    print(f"Best r2 =  {best_r2:.4f}")
    coef = best_lasso.coef_
    inter = best_lasso.intercept_
    print(f"Best Intercept {inter:.4f}")
    for i in range(len(coef)):
        print(f"theta_{i} \t {coef[i]:.4f}")

def transform(X):
    X_new=X
    col0 = X_new[glb.feat_dic[0]]
    col2 = X_new[glb.feat_dic[0]]
    Age = X_new[glb.feat_dic[7]]
    col4 = X_new[glb.feat_dic[4]]
    col5 = X_new[glb.feat_dic[5]]
    col6 = X_new[glb.feat_dic[6]]
    col1_2 = X_new[glb.feat_dic[1]]
    col_456 = []
    col_log0 = []
    col_2sqrt = []
    drop_index=[4,5,6]
    for i in drop_index:
        X_new=X_new.drop(glb.feat_dic[i], axis=1)


    for i in range(len(Age)):
        x0=col0[i]
        x2=col2[i]
        x4 = col4[i]
        x5 = col5[i]
        x6 = col6[i]
        Age[i] = math.log(float(Age[i]))
        col_456.append(math.pow(x4*x5*x6, 0.5))
        col4[i] = math.pow(x4, 1.66)
        col1_2[i] = math.pow(col1_2[i], 2)
        col_log0.append(math.log(x0))

    X_new["log(Age_day_)"] = Age
    X_new["Superplasticizer_component5__kgInAM_3Mixture_**1.66"] = col4
    X_new["BlastFurnaceSlag_component2__kgInAM_3Mixture_**2"] = col1_2
    X_new["x4*x5*x6"] = col_456
    X_new["logx0"] = col_log0

    return X_new



def Q4_explore():
    df_train = pd.read_csv('Data/train.csv')

    Y=df_train[glb.target_name]
    X=df_train.drop(glb.target_name, axis=1)

    X_new = transform(X)


    step=0.0001
    grid=[]
    for i in range(1, int(0.01/step)):
        grid.append(i*step)

    #regularized_lasso(X_new,Y,grid)
    regularized_net(X_new, Y, grid)
    #regularized_ridge(X_new, Y, grid)
    #regularized_llar(X_new, Y,grid)

Q4_explore()


#Provisional alpha = 0.07
# Best alpha = 0.07
# Best r2 =  0.6206
# Best Intercept 35.5792
# theta_0 	 10.9804
# theta_1 	 7.4779
# theta_2 	 4.3624
# theta_3 	 -5.1628
# theta_4 	 0.9840
# theta_5 	 -0.0421
# theta_6 	 0.0000
# theta_7 	 7.3513

#Dropping 4,5,6
# Best alpha = 0.02
# Best r2 =  0.6207
# Best Intercept 35.5792
# theta_0 	 11.4657
# theta_1 	 7.9937
# theta_2 	 4.9999
# theta_3 	 -5.7509
# theta_4 	 7.4669


#Log of time , dropping 4,5,6
# Best alpha = 0.009
# Best r2 =  0.8239
# Best Intercept 35.5792
# theta_0 	 11.4617
# theta_1 	 7.6562
# theta_2 	 4.0567
# theta_3 	 -5.6303
# theta_4 	 10.4652


# GNuplot fits

# Var Col 5 = to the power of 1.6
# Var Col 6 to the power of -5 baaad as well
# Var 7 is fucked

# Log of Time, Drop 5,6, Col4**1.6
# Best alpha = 0.013000000000000001
# Best r2 =  0.8260
# Best Intercept 35.5792
# theta_0 	 11.8306
# theta_1 	 7.9714
# theta_2 	 4.3328
# theta_3 	 -6.2799
# theta_4 	 10.5331
# theta_5 	 -1.1547


#Fitted Column 2 with 0.50*$2 -0.0014*$2**2
#Alpha wants to go to 0 here
# Best alpha = 0.001
# Best r2 =  0.8304
# Best Intercept 35.5792
# theta_0 	 11.9442
# theta_1 	 11.3698
# theta_2 	 4.4120
# theta_3 	 -6.4150
# theta_4 	 10.5138
# theta_5 	 -1.5071
# theta_6 	 -3.5054


# Using Ridge from now on
# Best alpha = 0.0099
# Best r2 =  0.8304
# Best Intercept 35.5792
# theta_0 	 11.9484
# theta_1 	 11.3904
# theta_2 	 4.4158
# theta_3 	 -6.4185
# theta_4 	 10.5149
# theta_5 	 -1.5130
# theta_6 	 -3.5230
