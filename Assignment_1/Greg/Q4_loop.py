import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from tools import *
import math

def HasD(x, y, w2=1):
# Implementation of Hassanat Distance
    HasD=0

    y[1]=w2*y[1]
    x[1]=w2*x[1]

    for i in range(0, len(x)):
        min_xy=min(x[i], y[i])
        max_xy=max(x[i], y[i])
        if min_xy >= 0:
            HasD+= 1 - (1+min_xy)/(1+max_xy)
        else:
            abs_min=abs(min_xy)
            HasD+= 1 -(1+min_xy+abs_min)/(1+max_xy+abs_min)

    return HasD

def Lorentz_dist(x, y):

    LD=0
    for i in range(0, len(x)):
        LD += math.log(1+abs(x[i]-y[i]))

    return LD

def SCSD(x, y):
    Chi=0

    for i in range(0, len(x)):
        dxy = x[i]-y[i]
        Chi+= dxy*dxy/abs(x[i]+y[i])

    return Chi

def Canberra(x,y):
    canD=0

    for i in range(0, len(x)):
        dxy = x[i]-y[i]
        canD+=abs(dxy)/(abs(x[i])+abs(y[i]))

    return canD



def Q4_loop():
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

    df_shuffled = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

    scaler=StandardScaler()
    X=df_shuffled[Global.columns_names]
    X_scaled=scaler.fit_transform(X)
    Y=df_shuffled[Global.label_name]

    print("k\tw2\terr\tstd_dev")
    k_range=range(15, 21, 1)
    w2_step=0.1
    w2_range=range(0, int(2/w2_step))
    print(w2_range)
    err_min=0.16
    k_min=17
    w2_min=1

    for k in k_range:
        for w2_index in w2_range:
            w2=w2_index*w2_step
            knn = KNeighborsClassifier(n_neighbors=k,
                                       metric=HasD,
                                       metric_params={"w2":w2})

            scores = cross_val_score(knn, X, Y, cv=5, scoring='accuracy')
            avg_score = scores.mean()
            avg_err= 1-avg_score
            score_std = scores.std()

            if(avg_err<err_min):
                print(f"{k}\t{w2}\t{avg_err:,.4f}\t{score_std:,.4f}")





Q4_loop()

# x=[-0.5,-1]
# y=[0,-10]

# print(Lorentz_dist(x,y))
