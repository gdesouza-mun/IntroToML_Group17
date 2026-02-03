import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Q4_Knn_Classifiers (X_train, y_train, X_test,y_test,grid_point, model, Distance_metric):
    '''==========================================
    Training phase and Compute training error
    =========================================='''
    len_Train_Data = len(X_train)
    A =np.empty((len_Train_Data, len_Train_Data))
    A.fill(0)
    # adjust sNC data
    #y_train2 = y_train.reshape(-1, 1)
    #y_train2 = y_train.ravel()
    ##print(X_train.shape)
    #print("Before adjustment:")
    #X_train[y_train2 == 0] += 0.5
    #print(X_train.shape)
    
    if Distance_metric == 'Euclidean':
        # Distance - compute a haff matrix
        for j in range(0, len_Train_Data,1):
            for i in range(j+1, len_Train_Data,1):
                A[i, j] =((X_train[i, 0] - X_train[j, 0] )**2 + (X_train[i, 1] - X_train[j, 1])**2)**0.5     
    else:
        # Manhattan Distance - compute a haff matrix
        for j in range(0, len_Train_Data,1):
            for i in range(j+1, len_Train_Data,1):
                A[i, j] = abs(X_train[i, 0] - X_train[j, 0]) + abs(X_train[i, 1] - X_train[j, 1])     
    
    #matrix A is symmetric -> compute only half and mirror
    A = A + A.T     
    # Compute training error
    Train_Error_count = 0
    y_pred_train= np.empty((len_Train_Data, 2))
    for i in range(0, len_Train_Data,1):
        indices_sorted = np.argsort(A[:, i])
        topk_indices = indices_sorted[:model] 
        count_sDAT = np.sum(y_train[topk_indices] == 1)
        count_sNC = model - count_sDAT
        #Preduction
        if count_sDAT > count_sNC:
            y_pred_train[i,0] = 1  # Predict sDAT
        else:
            y_pred_train[i,0] = 0  # Predict sNC
        # Count errors
        if y_pred_train[i,0] != y_train[i]:
            Train_Error_count += 1
            y_pred_train[i,1] = 0  # Mark error
        else:
            y_pred_train[i,1] = 1  # Mark correct prediction

    Training_error = float(Train_Error_count) / float(len_Train_Data)
    
    '''==========================================
    Testing phase and Compute test error
    =========================================='''
    len_TestData = len(X_test)
    A_test =np.empty((len_Train_Data,len_TestData))
    if Distance_metric == 'Euclidean':
        for j in range(len_TestData):
            for i in range(len_Train_Data):
                A_test[i, j] = ((X_test[j, 0] - X_train[i, 0])**2 + (X_test[j, 1] - X_train[i, 1])**2)**0.5
    else:
        for j in range(len_TestData):
            for i in range(len_Train_Data):
                A_test[i, j] = abs(X_test[j, 0] - X_train[i, 0]) + abs(X_test[j, 1] - X_train[i, 1])
    # Compute test error
    Test_Error_count = 0
    # np_Test_result: first column - prediction;
    #  second column - error (0 if error, 1 if correct)
    y_pred_test = np.empty((len_TestData, 2))

    for j in range(len_TestData):
        indices_sorted = np.argsort(A_test[:,j])
        topk_indices = indices_sorted[:model]

        count_sDAT = np.sum(y_train[topk_indices] == 1)
        count_sNC = model - count_sDAT
        #Preduction
        if count_sDAT > (count_sNC+ AdjustFunction(X_test[j,0], X_test[j,1])):
            y_pred_test[j,0] = 1  # Predict sDAT
        else:
            y_pred_test[j,0] = 0  # Predict sNC
        # Count errors
        if y_pred_test[j,0] != y_test[j]:
            Test_Error_count += 1
            y_pred_test[j,1] = 0 
        else:
            y_pred_test[j,1] = 1
    Test_error = float(Test_Error_count) / float(len_TestData)
    '''==   ===========================================
    Generate classification boundary visualization
    =========================================='''
    
    X_grid = grid_point.values
    len_GridData = len(X_grid)  
    A_grid =np.empty((len_Train_Data,len_GridData))
    if Distance_metric == 'Euclidean':
        for j in range(len_GridData):
            for i in range(len_Train_Data):
                A_grid[i, j] = ((X_grid[j, 0] - X_train[i, 0])**2 + (X_grid[j, 1] - X_train[i, 1])**2)**0.5                 
    else:
        for j in range(len_GridData):
            for i in range(len_Train_Data):
                A_grid[i, j] = abs(X_grid[j, 0] - X_train[i, 0]) + abs(X_grid[j, 1] - X_train[i, 1])            
    # Predict grid points
    y_pred_grid = np.empty((len_GridData, 2))   
    for j in range(len_GridData):
        indices_sorted = np.argsort(A_grid[:,j])
        topk_indices = indices_sorted[:model]

        count_sDAT = np.sum(y_train[topk_indices] == 1)
        count_sNC = model - count_sDAT
        #Preduction
        if count_sDAT > count_sNC:
            y_pred_grid[j,0] = 1  # Predict sDAT
        else:
            y_pred_grid[j,0] = 0  # Predict sNC
      
    ''' Return training and test error '''

    return Training_error, Test_error, y_pred_test, y_pred_grid
def AdjustFunction(x1, x2):
    
    valid_val = x1 + x2
    if valid_val < 2:
        return 1
    elif valid_val < 3.0:
        return 4
    else:
        return 1
   
