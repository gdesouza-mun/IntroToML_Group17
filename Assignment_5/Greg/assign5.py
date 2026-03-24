import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics.pairwise import pairwise_distances



import os
import sys

class Global:
    data_name="mnist_train_data.npy"
    label_name="mnist_train_labels.npy"

    k_folds=5

def error_rate(y_pred, y_true, model_name=""):

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    error_rate = np.mean(y_pred != y_true)

    print("Err assessment for model: ", model_name)
    print(error_rate)


#Question 1


#Load data... loads the data, and Q1_results proves that we loaded according to instructions

def load_data(data_dir="data", train_per_sample=4860, subset_size=0):
    data_path = os.path.join(data_dir, Global.data_name)
    label_path= os.path.join(data_dir, Global.label_name)

    data_np = np.load(data_path)
    label_np = np.load(label_path)

    if subset_size:
        indices = np.arange(label_np.shape[0])
        np.random.shuffle(indices)
        subset_indices = indices[:subset_size]
        data_np = data_np[subset_indices]
        label_np = label_np[subset_indices]

    all_indices = np.arange(len(label_np))
    train_indices = []

    num_samples=train_per_sample
    for label_value in np.unique(label_np):
        label_indices = np.where(label_np == label_value)[0]
        chosen = np.random.choice(label_indices, num_samples, replace=False)
        train_indices.extend(chosen)


    train_indices = np.array(train_indices)
    test_indices = np.delete(all_indices, train_indices)

    x_train = data_np[train_indices]
    y_train = label_np[train_indices]

    x_test = data_np[test_indices]
    y_test = label_np[test_indices]

    return x_train, y_train, x_test, y_test

def Q1_results():
    x_train, y_train, x_test, y_test = load_data()

    print("Condition 1: 60 000 total entries between train/test split")
    print("train \t + test \t = total")
    print(f"{x_train.shape[0]} \t + {x_test.shape[0]} \t = {x_train.shape[0]+x_test.shape[0]}")

    print("\n \n Now let's check Criteria 2 and 3 - The train set is balanced, and the test set contains at least 10% of each class \n")

    print("Label \t Train \t Test \t Fraction")
    for value in range(0,10):
        value_train_counts = np.sum(y_train==value)
        value_test_counts = np.sum(y_test==value)
        frac = value_test_counts/(value_test_counts+value_train_counts)
        print(f"{value} \t {value_train_counts} \t {value_test_counts} \t {frac:.4f}")



def Q2_explore():
    x_train, y_train, _, _ = load_data("data")


    #vectorizing be columns (Fortran order 'F')
    x_train_vec = x_train.reshape(x_train.shape[0], -1, order='F')

    #k_list = [1, 3, 5, 10, 25, 50, 100, 150, 200]
    k_list = [1,3,5]

    param_dic={ "n_neighbors":k_list}

    dist_matrix = pairwise_distances(x_train_vec, metric='euclidean')
    kNN = KNeighborsClassifier(metric="precomputed")
    grid_search=GridSearchCV(kNN,
                               param_grid=param_dic, scoring='accuracy',
                               refit=False, return_train_score=True,
                               verbose=2, cv=5)

    grid_search.fit(dist_matrix, y_train)

    k_scores = grid_search.cv_results_["mean_test_score"]
    k_train_scores = grid_search.cv_results_["mean_train_score"]

    print("k_list")
    print(k_list)
    print("test scores")
    print(k_scores)
    print("train scores")
    print(k_train_scores)

def Q2_results():
    #From Q2_explore, k=3 is the best model
    x_train, y_train, x_test, y_test = load_data("data")

    x_train_vec = x_train.reshape(x_train.shape[0], -1, order='F')
    x_test_vec = x_test.reshape(x_test.shape[0], -1, order='F')

    dist_matrix = pairwise_distances(x_train_vec, metric='euclidean')
    kNN=KNeighborsClassifier(3, metric="precomputed")

    kNN.fit(dist_matrix, y_train)

    test_dist_matrix = pairwise_distances(x_test_vec, x_train_vec, metric="euclidean")
    y_pred = kNN.predict(test_dist_matrix)

    error_rate(y_pred, y_test, "kNN for k=3")



Q2_results()
