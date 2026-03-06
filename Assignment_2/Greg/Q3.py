import pandas as pd
import matplotlib.pyplot as plt
from dataVisu import glb
import math

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score


def Q3_gridsearch(X,Y, grid):
    lasso = Lasso()

    param_grid = {'alpha': grid}

    # scoring = { 'mse' : 'neg_mean_squared_error',
    #             'r2' : 'r2' }
    fold=10
    grid_search = GridSearchCV(
        estimator=lasso,
        param_grid=param_grid,
        cv=fold,
        scoring=["neg_mean_squared_error", "r2"],
        return_train_score=False,
        refit="r2")

    grid_search.fit(X,Y)

    best_idx = grid_search.best_index_
    best_alpha = grid_search.best_params_['alpha']
    best_r2 = grid_search.cv_results_['mean_test_r2'][best_idx]
    best_mse = -grid_search.cv_results_['mean_test_neg_mean_squared_error'][best_idx]

    sample_size = len(X)/fold
    RSE = glb.MSEtoRSE(best_mse, sample_size)

    print(" alpha \t r2 \t RSE")
    print(f"{best_alpha} \t {best_r2:.4} {RSE:.4}")

def Q3_explore():

    df_train = pd.read_csv('Data/train.csv')

    Y=df_train[glb.target_name]
    X=df_train.drop(glb.target_name, axis=1)

    #grid=[]
    # step=1
    # for ind in range(0, int(10/step)):
    #     grid.append(step+step*ind)
    grid=range(1,100)
    Q3_gridsearch(X, Y,  grid)

def Q3_results():
    df_train = pd.read_csv('Data/train.csv')

    Y=df_train[glb.target_name]
    X=df_train.drop(glb.target_name, axis=1)
    alpha=1

    lasso=Lasso(alpha=alpha)

    lasso.fit(X,Y)

    df_test = pd.read_csv('Data/test.csv')

    Y_test=df_test[glb.target_name]
    X_test=df_test.drop(glb.target_name, axis=1)

    Y_pred=lasso.predict(X_test)

    r2 = r2_score(Y_pred, Y_test)
    MSE = mean_squared_error(Y_pred, Y_test)
    sample_size = len(Y_test)
    RSE = glb.MSEtoRSE(MSE, sample_size)

    coef = lasso.coef_
    inter = lasso.intercept_

    print(" alpha \t r2 \t RSE")
    print(f"{alpha} \t {r2:.4} {RSE:.4}")
    print(f"======== Best Parameters ======")
    print(f"Intercept \t {inter:.4}")
    for i in range(len(coef)):
        print(f"theta_{i} \t {coef[i]:.4f}")

Q3_results()


#best alpha=1; 1 	 0.6198 11.0

#Best Resuts
#  alpha 	 r2 	 RSE
# 1 	 0.3151 10.02
