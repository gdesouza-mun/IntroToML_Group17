import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Q4_Knn_Classifiers import Q4_Knn_Classifiers
from matplotlib.ticker import LogLocator


'''   Plot data from train.sDAT.csv and train.sNC.csv   '''

# Load  train data
train_sDAT = pd.read_csv('Data/train.sDAT.csv', header=None, names=['X1_sDAT', 'X2_sDAT'])
train_sNC = pd.read_csv('Data/train.sNC.csv', header=None, names=['X1_sNC', 'X2_sNC'])

X_train = np.vstack([train_sNC.values, train_sDAT.values])
y_train = np.vstack([np.zeros((len(train_sNC),1)), np.ones((len(train_sDAT),1))])
#print(y_train)
# Load test data
test_sNC = pd.read_csv('Data/test.sNC.csv', header=None,names=['X1_sNC', 'X2_sNC'])
test_sDAT = pd.read_csv('Data/test.sDAT.csv', header=None,names=['X1_sDAT', 'X2_sDAT'])
X_test = np.vstack([test_sNC.values, test_sDAT.values])
y_test = np.vstack([np.zeros((len(test_sNC),1)), np.ones((len(test_sDAT),1))])


# Load grid points
grid_point = pd.read_csv('Data/2D_grid_points.csv', header=None, names=['X1', 'X2'])
X_grid = grid_point.values
# Plot Training error result
k = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]
#k=[30,50]
training_errors = []
test_errors = []
 # Create output directory if it doesn't exist
output_dir = 'kNN_Plots'
if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Plotting
for model in k:
    train_error, test_error,y_pred_test, y_pred_grid = Q4_Knn_Classifiers(X_train, y_train, X_test, y_test,grid_point, model,"Euclidean")
    training_errors.append(train_error)
    test_errors.append(test_error)

   
    fig, ax1 = plt.subplots(figsize=(9, 9) )
    
    ax1.scatter(train_sNC.iloc[:,0], train_sNC.iloc[:,1], marker='o', alpha=0.9, s=15, color='green', label='Train Dat: Cls 0 ')
    ax1.scatter(train_sDAT.iloc[:,0], train_sDAT.iloc[:,1], marker='o', alpha=0.9, s=15, color='blue', label='Train Dat: Cls 1')
    
    pred  = y_pred_test[:, 0]
    right = y_pred_test[:, 1]
    
    ax1.scatter(X_test[(right==1)&(pred==0), 0],
            X_test[(right==1)&(pred==0), 1],
            marker='+', color='green', label='Test Dat:Pred->Cls 0; True->Cls 0')

    ax1.scatter(X_test[(right==1)&(pred==1), 0],
            X_test[(right==1)&(pred==1), 1],
            marker='+', color='blue', label='Test Dat:Pred->Cls 1: True->Cls 1')

    ax1.scatter(X_test[(right==0)&(pred==0), 0],
            X_test[(right==0)&(pred==0), 1],
            marker='+', color='red', label='Test Dat:Pred->Cls 0; True->Cls 1')
    ax1.scatter(X_test[(right==0)&(pred==1), 0],
            X_test[(right==0)&(pred==1), 1],
            marker='+', color='purple', label='Test Dat:Pred->Cls 1; True->Cls 0')
    #ax1.scatter(test_sDAT.iloc[:,0], test_sDAT.iloc[:,1], marker='+', alpha=0.9, s=50, color='blue', label='Class 1 - sDAT Test')
    #ax1.scatter(test_sNC.iloc[:,0], test_sNC.iloc[:,1], marker='+', alpha=0.9, s=50, color='green', label='Class 0 - sNC Test')
    
    colors = ['blue' if y==1 else 'green' for y in y_pred_grid[:, 0]]

    ax1.scatter(X_grid[:,0], X_grid[:,1], marker='.', alpha=1, s=7, color=colors, label='Grid Points')

    ax1.set_xlabel('Feature X1')
    ax1.set_ylabel('Feature X2')
    ax1.set_xlim(0.9, 2.2)
    ax1.set_ylim(0.9, 2.2)
    ax1.set_title("Decision Boundary of KNN Classifier (k = {}, Euclidean Distance)\n " \
    "Training Error: {:.4f}, Test Error: {:.4f}".format(model,train_error, test_error))   

    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'Q4_Euclidean_kNN_k={model}.png'), dpi=150, bbox_inches='tight')
    plt.close()


fig, ax2 = plt.subplots(figsize=(9, 9) )

ax2.plot(k, training_errors, marker='o', linestyle='-', color='green',label=' Training Error ')
ax2.plot(k, test_errors, marker='x', linestyle='-', color='red',label=' Test Error ')
ax2.set_xlabel('Hyperparameter k')
ax2.set_ylabel('Training/Test Error rate')
ax2.set_title('KNN Classifier Model with Euclidean Distance \n \
Training/Test Error vs Hyperparameter k')
ax2.grid(True, alpha=0.3)
ax2.grid(True, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)
ax2.legend()
plt.savefig(os.path.join(output_dir, f'Q4_Euclidean_Performance_vs_K.png'), dpi=150, bbox_inches='tight')
plt.close()


''' ===============================================
Question 3: KNN Classifier with Euclidean Distance 
Error rate versus Model capacity (hyperparameter k)  
=============================================== '''

fig, ax3 = plt.subplots(figsize=(9, 9) )

inv_k = [1/x for x in k]
bayes_error = min(test_errors)

plt.xscale('log')

ax3.plot(inv_k, training_errors, marker='o', linestyle='-', color='green',label=' Training Error ')
ax3.plot(inv_k, test_errors, marker='x', linestyle='-', color='red',label=' Test Error ')
ax3.axhline(y=bayes_error, color='brown', linestyle='--', label='Bayes Error (bayes_error= {:.4f})'.format(bayes_error))
ax3.set_xlabel('Model Capacity (1/k)- Log Scale')
ax3.set_ylabel('Training/Test Error Rate')
ax3.set_title('KNN Classifier Model with Euclidean Distance \n \
Training and Test Error vs Model Capacity (log-scale)')
ax3.grid(True, alpha=0.3)
ax3.grid(True, color='gray', linestyle='--', linewidth=1.2, alpha=0.8)
ax3.legend()

ax = plt.gca()
ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=20))
ax.grid(True, which='both', linestyle='--', alpha=0.7)

plt.savefig(os.path.join(output_dir, f'Q4_Euclidean_Performance_vs_K_Log_Scale.png'), dpi=150, bbox_inches='tight')
plt.close()

print("Training Phase plot displayed.")
print(training_errors)
