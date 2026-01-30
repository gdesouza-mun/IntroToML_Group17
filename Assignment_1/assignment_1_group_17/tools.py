import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


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
