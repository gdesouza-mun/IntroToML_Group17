#  Assignment 4
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

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier


#So we access each model identically
def print_assessement(y_test, y_pred, model_name=""):
    # Calculate standard metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred) # Recall is equivalent to Sensitivity
    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    # Calculate Specificity using the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    print(f"\n--- Performance Metrics on Test Set for {model_name}---")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Sensitivity:       {sensitivity:.4f}")
    print(f"Specificity:       {specificity:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {sensitivity:.4f}")


class Global:
    train_pos_path="train.fdg_pet.pMCI.csv"
    train_neg_path="train.fdg_pet.sMCI.csv"

    test_pos_path="test.fdg_pet.pMCI.csv"
    test_neg_path="test.fdg_pet.sMCI.csv"

    neg_label=0
    pos_label=1

    feature_names=["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8",
                   "x9", "x10", "x11", "x12", "x13"]

    random_seed=17
    cv_folds=10
    main_score='balanced_accuracy'
    crit_list=["gini", "entropy", "log_loss"]


def load_data(data_dir="Data"):
    train_neg_path = os.path.join(data_dir, Global.train_neg_path)
    train_neg = pd.read_csv(train_neg_path, header=None)
    train_neg["y"] = Global.neg_label

    train_pos_path = os.path.join(data_dir, Global.train_pos_path)
    train_pos = pd.read_csv(train_pos_path, header=None)
    train_pos["y"] = Global.pos_label

    train_df = pd.concat([train_neg, train_pos], ignore_index=True)

    test_neg_path = os.path.join(data_dir, Global.test_neg_path)
    test_neg = pd.read_csv(test_neg_path, header=None)
    test_neg["y"] = Global.neg_label

    test_pos_path = os.path.join(data_dir, Global.test_pos_path)
    test_pos = pd.read_csv(test_pos_path, header=None)
    test_pos["y"] = Global.pos_label

    test_df = pd.concat([test_neg, test_pos], ignore_index=True)

    y_train=train_df["y"]
    X_train=train_df.drop(columns=["y"])

    y_test=test_df["y"]
    X_test=test_df.drop(columns=["y"])

    return X_train, y_train, X_test, y_test




####################################################################################
# Question 1
####################################################################################

def Q1_results():
    # Train decision tree based on gini or log loss CV
    # Retrain on best
    X_train, y_train, X_test, y_test = load_data()

    crit_dic = {"criterion": Global.crit_list}

    tree_class = DecisionTreeClassifier()
    grid_search = GridSearchCV(tree_class, crit_dic,
                             cv=Global.cv_folds, scoring=Global.main_score,
                             refit=True)

    grid_search.fit(X_train, y_train)
    best_crit=grid_search.best_params_["criterion"]
    mean_cv_scores = grid_search.cv_results_['mean_test_score']
    print("Mean Balanced Accuracy During cross validation:")
    for i in range(len(Global.crit_list)):
        print(Global.crit_list[i], f" : {mean_cv_scores[i]:.4f}")


    print("Best Criterion for the tree is: ", best_crit)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print_assessement(y_test, y_pred, "Decision Tree Classifier")


Q1_results()
