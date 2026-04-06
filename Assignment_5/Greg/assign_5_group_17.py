#  Assignment 5
#
#  Group 17:
#  Darlington Nkrumah, MUN ID 202492437, dknkrumah@mun.ca
#  Greg de Souza, MUN ID 2025225,  gdesouza@mun.ca
#  Xuan Toan Doan, MUN ID 202583882, txdoan@mun.ca


####################################################################################
# Imports
####################################################################################


import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


import os
import sys

# Globa Utilities

class Global:
    #Just saving data file names
    data_name="mnist_train_data.npy"
    label_name="mnist_train_labels.npy"


def error_rate(y_pred, y_true, model_name=""):
    #To evaluate every model similary
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    error_rate = np.mean(y_pred != y_true)

    print("Err assessment for model: ", model_name)
    print(error_rate)


####################################################################################
# Question 1
####################################################################################

def load_data(data_dir="data", train_per_sample=4860):
    '''
    load_data(
    data_dir -> Path to data (assuming names saved in Global)
    train_per_sample -> How many samples of each class to take for the
    training data? 4860 is the number that maximally satisfies the condition
    for Q1 (Label 5 has only 5400 elements, so 90% of that is 4860)
    '''
    data_path = os.path.join(data_dir, Global.data_name)
    label_path= os.path.join(data_dir, Global.label_name)

    #Load data to numpy array
    data_np = np.load(data_path)
    label_np = np.load(label_path)

    all_indices = np.arange(len(label_np))
    train_indices = []

    for label_value in np.unique(label_np): #For every class

        #Get indices that are of that class
        label_indices = np.where(label_np == label_value)[0]

        #Get a random choice of train_per_sample of that class for training
        chosen = np.random.choice(label_indices,
                                  train_per_sample, replace=False)
        train_indices.extend(chosen)


    train_indices = np.array(train_indices)
    #The testing data is the complement of the training data
    test_indices = np.delete(all_indices, train_indices)

    x_train = data_np[train_indices]
    y_train = label_np[train_indices]

    x_test = data_np[test_indices]
    y_test = label_np[test_indices]

    #return the data in 4 arrays
    return x_train, y_train, x_test, y_test

def Q1_results():
    '''
    Q1_results : Proves that calling load_data() satysfies the conditions
    of Q1.
    '''
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




####################################################################################
# Question 2
####################################################################################
def Q2_explore():
    '''
    Q2 explore is the grid search for the k of the kNN, the details
    on how I proceeded are on the report
    '''
    x_train, y_train, _, _ = load_data("data")

    #vectorizing be columns (Fortran order 'F')
    x_train_vec = x_train.reshape(x_train.shape[0], -1, order='F')

    #k_list = [1, 3, 5, 10, 25, 50, 100, 150, 200]
    k_list = [1,3,5]

    param_dic={ "n_neighbors":k_list}

    #Notably I'm pre computing the distances
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

def Q2_graph():
    '''
    Makes the graph of the results for the grid search based on the manually
    inserted data. Details on the report
    '''
    #Results for round one, samples=1000/Class
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
    #plt.show()
    print("Saving image as Q2_results.png")
    plt.savefig("Q2_results.png", dpi=300)


    # Results for round three, samples = 4860/class

def Q2_results():
    #From Q2_explore, k=3 is the best model

    x_train, y_train, x_test, y_test = load_data("data")

    x_train_vec = x_train.reshape(x_train.shape[0], -1, order='F')
    x_test_vec = x_test.reshape(x_test.shape[0], -1, order='F')

    #I just refit the best k on all data and asses it with Err
    dist_matrix = pairwise_distances(x_train_vec, metric='euclidean')
    kNN=KNeighborsClassifier(3, metric="precomputed")

    kNN.fit(dist_matrix, y_train)

    test_dist_matrix = pairwise_distances(x_test_vec, x_train_vec, metric="euclidean")
    y_pred = kNN.predict(test_dist_matrix)

    error_rate(y_pred, y_test, "kNN for k=3")


####################################################################################
# Question 3
####################################################################################

def Q3_explore():
    #Same logic as Q2_explore, a function to do the grid search
    x_train, y_train, _, _ = load_data("data")

    #vectorizing be columns (Fortran order 'F')
    x_train_vec = x_train.reshape(x_train.shape[0], -1, order='F')

    C_list = np.logspace(-1,1, 25)
    #C_list = np.linspace(6.7,7,10)
    #d_list = [2,3,4]
    d_list = [2]
    param_dic = {"C": C_list,
                 "degree": d_list}

    SVM = SVC(kernel='poly', cache_size=200, random_state=42)
    grid_search = GridSearchCV(SVM, param_grid = param_dic,
                               scoring='accuracy', refit=True,
                               verbose=2, cv=5, n_jobs=4)

    grid_search.fit(x_train_vec, y_train)

    best_idx = grid_search.best_index_
    C_best = grid_search.cv_results_['params'][best_idx]['C']
    d_best = grid_search.cv_results_['params'][best_idx]['degree']
    best_score = grid_search.best_score_

    print("C \t degree \t score")
    print(f'{C_best:.4f} \t {d_best} \t {best_score:.4f}')

def Q3_results():
    #From Q3_explore, we determined the best C and degree
    x_train, y_train, x_test, y_test = load_data("data")

    x_train_vec = x_train.reshape(x_train.shape[0], -1, order='F')
    x_test_vec = x_test.reshape(x_test.shape[0], -1, order='F')

    SVM = SVC(kernel='poly', cache_size=200, random_state=42,
              C=6.8129, degree=2)

    SVM.fit(x_train_vec, y_train)

    y_pred = SVM.predict(x_test_vec)

    error_rate(y_pred, y_test, "SVM poly for C=6.8129 degree=2")


####################################################################################
# Question 4
####################################################################################

#Q4 is a bit more elaborate, there maybe a better way to do it
#But given the time limit I just did something I know it works

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

class DynamicMLP(nn.Module):
    '''
    This is just a Torch Neural Network for a feedforward NN
    that takes L and K as arguments during initialization
    '''
    def __init__(self, L, K, input_dim=784, output_dim=10):
        super().__init__()
        layers=[]
        current_dim = input_dim

        #Creates L hidden layers with ReLU activation
        for l in range(L):
            layers.append(nn.Linear(current_dim, K))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            current_dim = K


        #Plus the output layer
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TorchClassifier(BaseEstimator, ClassifierMixin):
    '''
    I then wrap the torch NN with a scikit estimator framework
    so I can use Grid Search CV with L an K
    '''

    def __init__(self, L=1, K=128, lr=0.001, epochs=5, batch_size=64):
        #Set the hyperparameters
        self.L=L
        self.K=L
        self.lr = lr
        self.epochs = epochs
        self.model = None
        self.batch_size = batch_size

    def toTensor_data(self, X):
        #The tensors will be normalized for regularity
        X_tensor = torch.tensor(X, dtype=torch.float32)/255.0
        return X_tensor

    def fit(self, X, y):
        #The fit method is required for the Grid Search

        self.model = DynamicMLP(L=self.L, K=self.K).to(device)

        X_tensor = self.toTensor_data(X)
        y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        #A basic training loop
        self.model.train()
        for epochs in range(self.epochs):
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        #The predict is the other necessary method for GridSeachCV
        self.model.eval()
        with torch.no_grad():
            X_tensor = self.toTensor_data(X).to(device)
            outputs = self.model(X_tensor)
            return torch.argmax(outputs, dim=1).cpu().numpy()


def Q4_explore(save=False, assess=False):
    #With those two classes I can just repeat the code for the previous
    #explorations

    x_train, y_train, x_test, y_test = load_data("data")

    x_train_vec = x_train.reshape(x_train.shape[0], -1, order='F')


    x_train_vec, y_train = shuffle(x_train_vec, y_train, random_state=42)

    # param_grid = {
    #     'L': [1,2,3,4,8,16],
    #     'K': [64, 128, 256, 512,784]
    # }

    param_grid = {
        'L': [7, 15],
        'K': [512, 784, 1024],
        'batch_size': [64],
        'lr': [5e-4]
    }


    n_jobs=1

    # print(device)
    # print(n_jobs)

    clf = TorchClassifier(epochs=30)
    grid = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy',
                        n_jobs=n_jobs, refit=True, verbose=2)

    grid.fit(x_train_vec, y_train)

    best_estimator = grid.best_estimator_
    L_best = best_estimator.L
    K_best = best_estimator.K
    bs_best = best_estimator.batch_size
    lr_best = best_estimator.lr

    print(f"Best L: {L_best}")
    print(f"Best K: {K_best}")
    print(f"Best batch size: {bs_best}")
    print(f"Best lr: {lr_best} \n")
    print(f"Best score: {grid.best_score_}")

    if assess:
        x_test_vec = x_test.reshape(x_test.shape[0], -1, order='F')
        y_pred = best_estimator.predict(x_test_vec)
        error_rate(y_pred, y_test, "Best MLP")

    #One difference here, I'm just saving the best model as is
    if save:
        save_name = f"best_Q4_L{L_best}_K{K_best}.pth"
        print("\n Saving the best perceptron as:", save_name)

        full_state = {
            "L": L_best,
            "K": K_best,
            'state_dict': best_estimator.model.state_dict()
        }

        torch.save(full_state, save_name)

#Q4_explore(save=True, assess=True)

def Q4_results():

    _, _, x_test, y_test = load_data("data")

    x_test_vec = x_test.reshape(x_test.shape[0], -1, order='F')
    X_test_tensor = torch.tensor(x_test_vec, dtype=torch.float32)/255.0

    #And I load the best model to get the predictions for the assessment
    saved_parameters = torch.load('Models/Q4_results.pth')
    saved_model = DynamicMLP(L=saved_parameters['L'],
                             K=saved_parameters['K'])
    print(f"Loading trained model with L={saved_parameters['L']} and K={saved_parameters['K']}")
    saved_model.load_state_dict(saved_parameters['state_dict'])
    saved_model.eval()

    with torch.no_grad():
        logits = saved_model(X_test_tensor)
        y_pred_tensor = torch.argmax(logits, dim=1)

    y_pred = y_pred_tensor.numpy()
    error_rate(y_pred, y_test, "Best MLP Model")




####################################################################################
# Question 5
####################################################################################

class SmallCNN(nn.Module):
    '''
    This is a small convolutional neural network, that will
    make predictions based on 28x28 images as input.
    '''
    def __init__(self, num_classes=10):
        super(SmallCNN, self).__init__()
        #First layer - Apply a 3x3 convolution going from 1 to 32 channels
        #So now, instead of 1,28x28 our data becomes 32x28,28

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        #Batch norm just makes sure the parameters don't explode
        self.bn1 = nn.BatchNorm2d(32)

        #Same thing, but going from 32 to 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        #Max pools(2,2) for each 4 pixel, it picks the max pixel, reducing the size
        #Now our image is 64 channels and 7x7
        self.pool = nn.MaxPool2d(2, 2)

        #Randomly sets 0.3 parameters to 0, so the network learns alternate paths
        self.dropout1 = nn.Dropout2d(0.3)

        #We pass through two linear layes to reduce from 64x7x7 to 10 labels
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        #We just call the network in order relu-ing where necessary
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def classifyHandwrittenDigits(Xtest, data_dir, model_path="Models/Q5_best.pth"):
    mean=0.1307
    std=0.3081

    Xtest_tensor = torch.from_numpy(Xtest.astype(np.float32)/255.0).unsqueeze(1)

    x_normalized = (Xtest_tensor - mean)/std

    saved_parameters=torch.load(model_path)
    model=SmallCNN()
    model.load_state_dict(saved_parameters['state_dict'])

    output = model(Xtest_tensor)

    y = torch.argmax(output, dim=1)

    return y






#########################################################################################
# Calls to generate the results
#########################################################################################
if __name__=="__main__":
    print("====== Q1_results() =========")
    Q1_results()
    print("====== Q2_results() =========")
    Q2_results()
    print("====== Q3_results() =========")
    Q3_results()
    print("====== Q4_results() =========")
    Q4_results()
    try:
        print("Trying to call classifyHandwrittenDigits")
        ytest = classifyHandwrittenDigits(Xtest, data_dir)
    except:
        print("Arguments for classifyHandwrittenDigits undefined")
