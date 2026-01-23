import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tools import *


def Q3():
    #Labeling the data Columns

    #Importing and labeling training data
    df_train_sNC=pd.read_csv('Data/train.sNC.csv', header=None)
    df_train_sNC.columns=Global.columns_names
    df_train_sNC[Global.label_name]=Global.sNC_label

    df_train_sDAT=pd.read_csv('Data/train.sDAT.csv', header=None)
    df_train_sDAT.columns=Global.columns_names
    df_train_sDAT[Global.label_name]=Global.sDAT_label

    #joining all the training data
    df_train=pd.concat([df_train_sNC, df_train_sDAT], axis=0, ignore_index=True)

    #Importing and labeling testing data
    df_test_sNC=pd.read_csv('Data/test.sNC.csv', header=None)
    df_test_sNC.columns=Global.columns_names
    df_test_sNC[Global.label_name]=Global.sNC_label

    df_test_sDAT=pd.read_csv('Data/test.sDAT.csv', header=None)
    df_test_sDAT.columns=Global.columns_names
    df_test_sDAT[Global.label_name]=Global.sDAT_label

    #joining all the testing data
    df_test=pd.concat([df_test_sNC, df_test_sDAT], axis=0, ignore_index=True)


    k_values=range(1, 100)
    array_1overk=[]
    array_test_err=[]
    array_train_err=[]
    for k in k_values:
        knn, scaler, err_train=train_kNN(df_train, k, metr='euclidean')
        err_test=test_kNN(df_test, knn, scaler)

        array_train_err.append(err_train)
        array_test_err.append(err_test)
        array_1overk.append(float(1/k))

    #rint(array_1overk, array_test_err)

    extra_array=array_1overk
    plt.plot(extra_array, array_test_err, marker='x', linestyle='-', label='Testing Error')
    plt.plot(array_1overk, array_train_err, marker='o', linestyle='-', label='Training Error')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='lower left')
    plt.show()



Q3()
