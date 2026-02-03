import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from tools import *
import math

def dist2(x,y):

    dx=x[0]-y[0]
    dy=x[1]-y[1]
    return math.sqrt(dx*dx + dy*dy)

def ToanMetric(x,y,w, cx, cy):

    center=[cx,cy]
    d2=dist2(x,y)
    # if x[0]+x[1] > 2 and x[0]+x[1]<3:
    #     d2=d2*w
    # if y[0]+y[1] > 2 and y[0]+y[1]<3:
    #     d2=d2*w

    e1=math.exp(w*dist2(x, center))
    e2=math.exp(w*dist2(y, center))

    return dist2(e1*x,e2*y)





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


    X=df_shuffled[Global.columns_names]
    X_scaled=X
    Y=df_shuffled[Global.label_name]
    k=24
    w=0.04
    min_error=0.168
    cx_start=0
    cy_start=0
    c_step=0.1
    center_range=range(0, round(2.2/c_step))

    for cx_index in center_range:
        cx=cx_start+cx_index*c_step
        for cy_index in center_range:
            cy=cy_start+cy_index*c_step
            knn = KNeighborsClassifier(n_neighbors=k,
                                   metric=ToanMetric,
                                   metric_params={"w":w, "cx":cx, "cy":cy},
                                   weights='distance')

            scores = cross_val_score(knn, X, Y, cv=5, scoring='accuracy')
            avg_score = scores.mean()
            avg_err= 1-avg_score
            score_std = scores.std()
            if avg_err<min_error:
                min_error=avg_err
                print(f"{cx}\t{cy}\t{avg_err:,.4f}\t{score_std:,.4f}")



        #Exp
        #   24	0.04	0.1676	0.0189




Q4_loop()

# x=[-0.5,-1]
# y=[0,-10]

# print(Lorentz_dist(x,y))
