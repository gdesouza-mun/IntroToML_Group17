import pandas as pd
import numpy as np
import os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


def predictCompressiveStrength(Xtest, data_dir):
    """
    Returns a vector of predictions of real number values,
    corresponding to each of the N_test features vectors in Xtest.

    Parameters:
    -----------
    Xtest : np.array or pd.DataFrame
        N_test x 8 matrix of test feature vectors
    data_dir : str
        Full path to the folder containing train.csv

    Returns:
    --------
    y_pred : np.array
        Vector of predicted compressive strength values
    """

    # 1. Construct the path to the training file
    train_path = os.path.join(data_dir, 'train.csv')

    # 2. Load the training data
    # We use the provided training data to train our 'best' model on the fly
    df_train = pd.read_csv(train_path)

    # 3. Prepare Training Data
    X_train = df_train.iloc[:, :-1]  # All columns except target
    y_train = df_train.iloc[:, -1]  # Target: Compressive Strength

    # 4. Initialize the 'Best' Model
    # HistGradientBoostingRegressor is chosen for its state-of-the-art performance
    # on tabular data. It handles non-linearities and interactions naturally.
    # We use conservative but robust hyperparameters.
    model = HistGradientBoostingRegressor(
        learning_rate=0.1,
        max_iter=500,  # Sufficient trees to converge
        max_depth=None,  # Allow trees to grow (regularized by min_samples_leaf)
        min_samples_leaf=20,  # Prevents overfitting to noise
        l2_regularization=0.1,  # Slight regularization
        random_state=42
    )

    # 5. Train the Model
    model.fit(X_train, y_train)

    # 6. Generate Predictions on the Input Test Set
    # Ensure Xtest is in the same format/order as X_train if it's a DataFrame
    # If Xtest is a numpy array, the model will handle it based on column index
    y_pred = model.predict(Xtest)

    return y_pred



df_test = pd.read_csv('test.csv')
Y=df_test['ConcreteCompressiveStrength_MPa_Megapascals_']
Xtest=(df_test.drop('ConcreteCompressiveStrength_MPa_Megapascals_', axis=1)).values

Ypred = predictCompressiveStrength(Xtest, '')

r2 = r2_score(Y, Ypred)

print(r2)
