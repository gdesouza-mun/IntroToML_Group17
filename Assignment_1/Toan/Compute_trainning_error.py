import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def compute_training_error  (df_sDAT, df_sNC, model):
    '''   Compute training error given sDAT and sNC dataframes and a model   '''
    len_Data = len(df_sDAT) + len(df_sNC)
    A =np.empty((len_Data, len_Data))
    np_sDAT = df_sDAT.to_numpy()
    np_sNC = df_sNC.to_numpy()
    # Distance from sDAT to sDAT
    for i in range(0, len(df_sDAT),1):
        for j in range(0, len(df_sDAT),1):
            A[i, j] =((np_sDAT[i, 0] - np_sDAT[j, 0] )**2 + (np_sDAT[i, 1] - np_sDAT[j, 1])**2)**0.5     
    
    # Distance from sDAT to sNC
    for i in range(0, len(df_sDAT), 1):
        for j in range(len(df_sDAT), len_Data, 1):
            A[i, j] = ((np_sDAT[i, 0] - np_sNC[j - len(df_sDAT), 0])**2 + (np_sDAT[i, 1] - np_sNC[j - len(df_sDAT), 1])**2)**0.5
    
    # Distance from sNC to sDAT
    for i in range(len(df_sDAT), len_Data, 1):
        for j in range(0, len(df_sDAT), 1):
            A[i, j] = ((np_sNC[i - len(df_sDAT), 0] - np_sDAT[j, 0])**2 + (np_sNC[i - len(df_sDAT), 1] - np_sDAT[j, 1])**2)**0.5
    # Distance from sNC to sNC
    for i in range(len(df_sDAT),len_Data,1):
        for j in range(len(df_sDAT),len_Data,1):
            A[i, j] = ((np_sNC[i - len(df_sDAT), 0] - np_sNC[j - len(df_sDAT), 0] )**2 + (np_sNC[i - len(df_sDAT), 1] - np_sNC[j - len(df_sDAT), 1])**2)**0.5
    # Compute training error
    Error_count = 0
    np_Training_result = np.empty((len_Data, 1))
    for i in range(0, len_Data,1):
        indices_sorted = np.argsort(A[:, i])
        topk_indices = indices_sorted[:model]
        count_sDAT = np.sum(topk_indices < len(df_sDAT))
        count_sNC = model - count_sDAT
        if count_sDAT > count_sNC:
            np_Training_result[i] = 1  # Predict sDAT
        else:
            np_Training_result[i] = 0  # Predict sNC
        # Count errors
        if i < len(df_sDAT):
            if np_Training_result[i] != 1:
                Error_count += 1       
        else:
            if np_Training_result[i] != 0:
                Error_count += 1
    Training_error = float(Error_count) / float(len_Data)
    return Training_error, np_Training_result