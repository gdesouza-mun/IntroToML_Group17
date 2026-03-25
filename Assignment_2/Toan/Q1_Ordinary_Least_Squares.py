from xml.parsers.expat import errors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def OLS_V (X_train, y_train):
    
    '''==========================================
    Validation method 
    =========================================='''
    #Split data into training, validation and test sub  sets
    X_train = np.array(X_train,dtype=float)
    y_train = np.array(y_train,dtype=float)
    X1_train = X_train[:480, :]
    y1_train = y_train[:480, :]
    X1_val = X_train[480:680, :]
    y1_val = y_train[480:680,:]
    X1_test =X_train[680:, :]
    y1_test = y_train[680:,:]
    # Compute theta using the normal equation
    X1_train = np.vstack((X1_train, X1_val))
    y1_train=np.vstack((y1_train, y1_val))
    theta = np.linalg.inv(X1_train.T @ X1_train) @ X1_train.T @ y1_train

    # Compute training error
    y1_pred_train = X1_train @ theta
    err = np.mean((y1_pred_train - y1_train) ** 2)

    # Compute test error
    y1_pred_test = X1_test @ theta
    Err = np.sqrt(np.mean((y1_pred_test - y1_test) ** 2))
    #compute Residual Sum of Squares (RSS)
    RSS = np.sum((y1_test - y1_pred_test) ** 2)
    # Compute Residual Standerd Error (RSE)
    RSE = np.sqrt(RSS / (len(y1_test) - len(theta)-1))
    # Compute R-squared
    mean_y1_test = np.mean(y1_test)
    TSS = np.sum((y1_test - mean_y1_test) ** 2)
    R_squared = 1 - (RSS / TSS) 
    
    return Err, RSE,R_squared, theta
def OLS_CV (X_train, y_train, k=5):
    '''==========================================
    k-fold CV method 
    =========================================='''
    X_train = np.array(X_train,dtype=float)
    y_train = np.array(y_train,dtype=float)
    fold_size = len(X_train) // k
    errors = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i != k - 1 else len(X_train)
        X_val = X_train[start:end]
        y_val = y_train[start:end]
        X_train_fold = np.concatenate((X_train[:start], X_train[end:]), axis=0)
        y_train_fold = np.concatenate((y_train[:start], y_train[end:]), axis=0)
        # Compute theta and errors for the current fold
        theta= np.linalg.inv(X_train_fold.T @ X_train_fold) @ X_train_fold.T @ y_train_fold
        Err= np.sqrt(np.mean((X_val @ theta - y_val) ** 2))
        RSS = np.sum((y_val - X_val @ theta) ** 2)
        RSE = np.sqrt(RSS / (len(y_val) - len(theta)-1))
        TSS = np.sum((y_val - np.mean(y_val)) ** 2)
        R_squared = 1 - (RSS / TSS)
        
        errors.append((Err,RSE,R_squared))
    final_theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    mean_errors = np.mean(errors, axis=0)
    return mean_errors[0],mean_errors[1],mean_errors[2], final_theta 