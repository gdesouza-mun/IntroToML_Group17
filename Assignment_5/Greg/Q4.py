from assign5 import *


import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin


class DynamicMLP(nn.Module):
    def __init__(self, L, K, input_dim=784, output_dim=10):
        super().__init__()
        layers=[]
        current_dim = input_dim

        for _ in range(L):
            layers.append(nn.Linear(current_dim, K))
            layers.append(nn.ReLU())
            current_dim = K


        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, L=1, K=128, lr=0.001, epochs=5):
        self.L=L
        self.K=L
        self.lr = lr
        self.epochs = epochs
        self.model = None

    def toTensor_data(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)/255.0
        return X_tensor
    def fit(self, X, y):
        X_tensor = self.toTensor_data(X)
        y_tensor = torch.tensor(y, dtype=torch.long)

        self.model = DynamicMLP(L=self.L, K=self.K)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epochs in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = self.toTensor_data(X)
            outputs = self.model(X_tensor)
            return torch.argmax(outputs, dim=1).numpy()


def Q4_explore():
    x_train, y_train, x_test, y_test = load_data("data")

    x_train_vec = x_train.reshape(x_train.shape[0], -1, order='F')
    x_test_vec = x_test.reshape(x_test.shape[0], -1, order='F')

    param_grid = {
        'L': [2,4,8,16],
        'K': [64, 128, 256,512]
    }

    clf = TorchClassifier(epochs=15)
    grid = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy',
                        n_jobs=8, refit=True, verbose=2)

    grid.fit(x_train_vec, y_train)

    print(f"Best Params: {grid.best_params_}")
    print(f"Best Score: {grid.best_score_} \n")

    best_estimator = grid.best_estimator_
    y_pred = best_estimator(x_test_vec)
    error_rate(y_pred, y_test, "Best MLP Model")

Q4_explore()
