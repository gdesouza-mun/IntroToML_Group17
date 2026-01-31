import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score



class Global:
    #Labels and Graphs according o assignment instructions
    sNC_label=0
    sNC_color='green'

    sDAT_label=1
    sDAT_color='blue'

    train_marker='o'
    test_marker='x'

    columns_names=['x1','x2']
    label_name='Target'

    sNC_train_path="Data/train.sNC.csv"
    sDAT_train_path="Data/train.sDAT.csv"

    sNC_test_path="Data/test.sNC.csv"
    sDAT_test_path="Data/test.sDAT.csv"



def HasD(x, y):
# Implementation of Hassanat Distance
    HasD=0

    for i in range(0, len(x)):
        min_xy=min(x[i], y[i])
        max_xy=max(x[i], y[i])
        if min_xy >= 0:
            HasD+= 1 - (1+min_xy)/(1+max_xy)
        else:
            abs_min=abs(min_xy)
            HasD+= 1 -(1+min_xy+abs_min)/(1+max_xy+abs_min)

    return HasD
