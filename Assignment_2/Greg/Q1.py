import pandas as pd
import matplotlib.pyplot as plt
from dataVisu import glb
import math
import statistics

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def Q1_val(X,Y):

    X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2, random_state=10)

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

    scoring = ["r2", "neg_mean_squared_error"]
    lin_model = LinearRegression()
    folds=10
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




Q1_results()
