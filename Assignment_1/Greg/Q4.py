import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from tools import *

def Q4_exploration():
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

    df_all = pd.concat([df_test, df_train], axis=0, ignore_index=True)


    #Get Randomized data
    df_shuffled = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    scaler=StandardScaler()
    X=df_shuffled[Global.columns_names]
    X_scaled=scaler.fit_transform(X)
    Y=df_shuffled[Global.label_name]

    k_range=range(10, 100, 2)

    w2_step=0.05
    w2_range=range(1, int(2/w2_step))

    w3_step=0.1
    w3_range=range(0, int(4/w3_step))

    p_values=[1, 2, 3, 4, 5, 6, 8, 10, 100]

    min_err = 1
    p_min = 2
    w2_min=1
    w3_min=1
    k_min=30

    for k in k_range:
        print(f"in k={k} \n")
        for p in p_values:
            for w2_index in w2_range:
                w2 = w2_index*w2_step
                for w3_index in w3_range:
                    w3 = w3_index*w3_step

                    knn = KNeighborsClassifier(n_neighbors=5,
                                               metric=custom_metric,
                                               metric_params={'w2': w2, 'w3': w3, 'power': p})

                    scores = cross_val_score(knn, X, Y, cv=5, scoring='accuracy')

                    avg_score = scores.mean()
                    avg_err= 1-avg_score
                    score_std = scores.std()

                    print(str(k) + "\t" + str(round(w2, 4)) + "\t" + str(round(w3, 4)) + "\t"+str(p) + "\t"+
                          str(round(avg_err, 5)) + "\t" + str(round(score_std, 5)) )

                    if(avg_err<min_err):
                        min_err = avg_err
                        p_min=p
                        w2_min=w2
                        w3_min=w3
                        k_min=k

    print(f"Minimum errror at: k={k_min} p={p_min} w2={w2_min} w3={w3_min} with avg_err={min_err}")


def custom_metric(x, y, w2, w3, power):

    dx = abs(x[0]-y[0])
    dy = abs(x[1]-y[1])

    met = pow(dx*dx, power) + pow(w2*dy*dy, power) + pow(w3*dx*dy, power)
    met = pow(met, 1/power)

    return met



Q4_exploration()
