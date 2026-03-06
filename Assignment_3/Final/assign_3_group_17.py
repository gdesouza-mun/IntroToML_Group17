#  Assignment 3
#
#  Group 17:
#  Darlington Nkrumah, MUN ID 202492437, dknkrumah@mun.ca
#  Greg de Souza, MUN ID 2025225,  gdesouza@mun.ca
#  Xuan Toan Doan, MUN ID 202583882, txdoan@mun.ca


####################################################################################
# Imports
####################################################################################
import sys
import os
import math

import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from sklearn.impute import SimpleImputer  # <-- Added import for handling NaNs

# Global Utilities

class Global:
    train_snc_path="Data/train.fdg_pet.sNC.csv"
    train_sdat_path="Data/train.fdg_pet.sDAT.csv"

    test_snc_path="Data/test.fdg_pet.sNC.csv"
    test_sdat_path="Data/test.fdg_pet.sDAT.csv"

    snc_label=0
    sdat_label=1

    feature_names=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
                   "x10", "x11", "x12", "x13", "x14"]

def generate_logspace(start, stop, num=50, endpoint=True, base=10.0):
    if num <= 0:
        return []

    # 1. Create a linear space (like numpy.linspace) for the exponents
    if endpoint:
        # Include the stop value in the linear range
        step = (stop - start) / (num - 1) if num > 1 else 0
        linear_space = [start + step * i for i in range(num)]
    else:
        # Exclude the stop value
        step = (stop - start) / num
        linear_space = [start + step * i for i in range(num)]

    # 2. Convert the linear space of exponents back to the original base
    log_space = [base ** exp for exp in linear_space]

    return log_space


####################################################################################
# Question 1
####################################################################################
def Q1_results():
    train_snc = pd.read_csv(Global.train_snc_path, header=None)
    train_snc.columns=Global.feature_names
    train_snc["y"]=Global.snc_label

    train_sdat = pd.read_csv(Global.train_sdat_path, header=None)
    train_sdat.columns=Global.feature_names
    train_sdat["y"]=Global.sdat_label

    train_df=pd.concat([train_snc, train_sdat], ignore_index=True)

    test_snc = pd.read_csv(Global.test_snc_path, header=None)
    test_snc.columns=Global.feature_names
    test_snc["y"]=Global.snc_label

    test_sdat = pd.read_csv(Global.test_sdat_path, header=None)
    test_sdat.columns=Global.feature_names
    test_sdat["y"]=Global.sdat_label

    test_df=pd.concat([test_snc, test_sdat], ignore_index=True)

    # y_train=train_df["y"]
    # X_train=train_df.drop["y"]

    print(train_df.drop(["y"]))

    # y_test=test_df["y"]
    # X_test=test_df.drop["y"]

    # C_logspace = generate_logspace(0.001, 1000)
    # print(C_logspace)


Q1_results()
