import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.svm import SVC
import matplotlib.pyplot as plt



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
    kNN = KNeighborsClassifier(metric="precomputed", random_state=42)
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

def Q2_graph():

    #Results for round one, samples=1000/class
    k1 = np.array([1, 3, 5, 10, 25, 50, 100, 150, 200])
    acc_test1 = np.array([0.9445, 0.9454, 0.9448, 0.9377, 0.9221, 0.9036, 0.8778, 0.8591, 0.8422])
    err_test1 = 1 - acc_test1
    acc_train1 = np.array([1. ,0.97315,  0.96395,  0.949375, 0.928375, 0.908425, 0.88195, 0.861125, 0.843775])
    err_train1 = 1 - acc_train1

    # Results for Round two, samples = 4840/class
    k2 = [1, 3, 5, 7, 9]
    acc_test2 = np.array([0.9627, 0.96296667, 0.96133333, 0.95953333, 0.95886667])
    err_test2 = 1 - acc_test2
    acc_train2 = np.array([1., 0.98076667, 0.97420833, 0.97053333, 0.9669])
    err_train2 = 1 - acc_train2

    min_err = min(err_test2)

    ftsize=18

    plt.xlabel('k', fontsize=ftsize+2)
    plt.xscale('log')
    plt.ylabel('Err', fontsize=ftsize+2)
    #plt.yscale('log')
    plt.title("5 fold Cross Validation test error for kNN")

    plt.plot(k1, err_test1, marker='s', linestyle=None, label='Test Error for 1000 Samples/Class',
             color='orange')
    plt.plot(k2, err_test2, marker='s', linestyle=None, label='Test Error for 4800 Samples/Class',
             color='blue')
    plt.axhline(y=min_err, color='black', linestyle = ':', label=f'Minimum Test Error: {min_err:.4f}')

    plt.legend(loc="upper left")
    plt.show()


    # Results for round three, samples = 4860/class

def Q2_results():
    #From Q2_explore, k=3 is the best model
    x_train, y_train, x_test, y_test = load_data("data")

    x_train_vec = x_train.reshape(x_train.shape[0], -1, order='F')
    x_test_vec = x_test.reshape(x_test.shape[0], -1, order='F')

    dist_matrix = pairwise_distances(x_train_vec, metric='euclidean')
    kNN=KNeighborsClassifier(3, metric="precomputed", random_state=42)

    kNN.fit(dist_matrix, y_train)

    test_dist_matrix = pairwise_distances(x_test_vec, x_train_vec, metric="euclidean")
    y_pred = kNN.predict(test_dist_matrix)

    error_rate(y_pred, y_test, "kNN for k=3")



# Question 3

#SVM same thing

def Q3_explore():
    x_train, y_train, _, _ = load_data("data", 1000, 15000)

    #vectorizing be columns (Fortran order 'F')
    x_train_vec = x_train.reshape(x_train.shape[0], -1, order='F')

    C_list = np.logspace(-3,3, 10)
    d_list = [2,3,4]
    param_dic = {"C": C_list,
                 "degree": d_list}

    SVM = SVC(kernel='poly', cache_size=400, random_state=42)
    grid_search = GridSearchCV(SVM, param_grid = param_dic,
                               scoring='accuracy', refit=True,
                               verbose=2, cv=5)

    grid_search.fit(x_train_vec, y_train)

    best_idx = grid_search.best_index_
    C_best = grid_search.cv_results_['params'][best_idx]['C']
    d_best = grid_search.cv_results_['params'][best_idx]['degree']
    best_score = grid_search.best_score_

    print("C \t degree \t score")
    print(f'{C_best:.4f} \t {d_best} \t {best_score:.4f}')



Q3_explore()
