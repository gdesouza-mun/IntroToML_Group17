import numpy as np
from assign_1_group_17 import diagnoseDAT


def Q4_test():

    # Generate a 100x2 matrix
    # low = 1.3, high = 2.2, size = (rows, columns)
    X = np.random.uniform(1, 2.2, size=(30, 2))
    data_path="Data"
    ytest=diagnoseDAT(X, data_path)

    print(ytest)

Q4_test()
