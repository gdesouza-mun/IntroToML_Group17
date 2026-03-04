import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def ytest_diagnoseDAT(Xtest, data_dir):
    """
    Returns a vector of predictions with elements "0" for sNC and "1" for sDAT,
    corresponding to each of the N_test features vectors in Xtest.
    """

    # 1. Construct file paths using the provided data_dir [cite: 106]
    train_snc_path = os.path.join(data_dir, 'train.fdg_pet.sNC.csv')
    train_sdat_path = os.path.join(data_dir, 'train.fdg_pet.sDAT.csv')
    test_snc_path = os.path.join(data_dir, 'test.fdg_pet.sNC.csv')
    test_sdat_path = os.path.join(data_dir, 'test.fdg_pet.sDAT.csv')

    # 2. Load all datasets, ensuring header=None to prevent feature name mismatches
    train_snc = pd.read_csv(train_snc_path, header=None)
    train_sdat = pd.read_csv(train_sdat_path, header=None)
    test_snc = pd.read_csv(test_snc_path, header=None)
    test_sdat = pd.read_csv(test_sdat_path, header=None)

    # 3. Combine ALL available data to train the strongest possible final model
    X_all_train = pd.concat([train_snc, train_sdat, test_snc, test_sdat], ignore_index=True)

    # Create target labels: "0" for sNC and "1" for sDAT [cite: 103]
    y_all_train = np.concatenate([
        np.zeros(len(train_snc)),
        np.ones(len(train_sdat)),
        np.zeros(len(test_snc)),
        np.ones(len(test_sdat))
    ])

    # 4. Handle Missing Data (NaNs)
    imputer = SimpleImputer(strategy='mean')
    X_all_train_imputed = imputer.fit_transform(X_all_train)

    # Ensure the Xtest passed into the function is also imputed
    Xtest_imputed = imputer.transform(Xtest)

    # 5. Scale the features (Crucial for the Polynomial kernel's performance)
    scaler = StandardScaler()
    X_all_train_scaled = scaler.fit_transform(X_all_train_imputed)

    # Scale the Xtest data using the same scaler
    Xtest_scaled = scaler.transform(Xtest_imputed)

    # 6. Initialize your BEST model (Polynomial: C=10, degree=3)
    best_model = SVC(kernel='poly', C=10, degree=3, random_state=42)

    # 7. Train the model on the fully combined, imputed, and scaled dataset
    best_model.fit(X_all_train_scaled, y_all_train)

    # 8. Predict on the provided Xtest
    predictions = best_model.predict(Xtest_scaled)

    return predictions