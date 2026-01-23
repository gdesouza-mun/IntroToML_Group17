import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

def import_grid():
    df=pd.read_csv('Grids/2D_grid_points.csv', header=None)
    df.columns=['x1', 'x2']

    X=df[['x1', 'x2']]

    X_scaled_array=scaler.transform(X)
    X_scaled=pd.DataFrame(X_scaled_array,
                          columns=X.columns,
                          index=X.index)

    return X_scaled

def import_scaled_data(datasetTag):

    if not (datasetTag=='train' or datasetTag=='test'):
        print("Invalid Data Set Tag")
        return

    df_sNC=pd.read_csv("Data/"+datasetTag+".sNC.csv", header=None)
    df_sNC['Target']=0
    df_sDAT=pd.read_csv("Data/"+datasetTag+".sDAT.csv", header=None)
    df_sDAT['Target']=1

    df=pd.concat([df_sNC, df_sDAT], axis=0, ignore_index=True)
    df.columns=['x1', 'x2', 'Target']

    X=df[['x1', 'x2']]
    if datasetTag=='train':
        X_scaled_array=scaler.fit_transform(X)
    elif datasetTag=='test':
        X_scaled_array=scaler.transform(X)

    X_scaled=pd.DataFrame(X_scaled_array,
                          columns=X.columns,
                          index=X.index)

    Y=df['Target']

    return X_scaled, Y

def kNN(k=1, m='euclidean', w='uniform'):

    X_scaled, Y=import_scaled_data('train')
    knn=KNeighborsClassifier(n_neighbors=k,
                             metric=str(m),
                             weights=str(w))

    knn.fit(X_scaled, Y)

    X_scaled_test, Y_test=import_scaled_data('test')

    X_grid=import_grid()

    predict_train=knn.predict(X_scaled)
    predict_test=knn.predict(X_scaled_test)
    predict_grid=knn.predict(X_grid)
    accuracy=knn.score(X_scaled_test, Y_test)

    return accuracy, predict_train, predict_test, predict_grid

def make_graphQ1(k=1, m='euclidean', w='uniform'):
    acc, Y1, Y2, Y3 = kNN(k, m, w)

    df=pd.read_csv('Grids/2D_grid_points.csv', header=None)
    df['Target']=Y3

    blue_group=df[df['Target']==1]
    green_group=df[df['Target']==0]


    df_test_sNC=pd.read_csv('Data/test.sNC.csv', header=None)
    df_test_sDAT=pd.read_csv('Data/test.sDAT.csv', header=None)
    df_train_sNC=pd.read_csv('Data/train.sNC.csv', header=None)
    df_train_sDAT=pd.read_csv('Data/train.sDAT.csv', header=None)

    n=7

    df_test_sNC=df_test_sNC.iloc[::n]
    df_test_sDAT=df_test_sDAT.iloc[::n]
    df_train_sNC=df_train_sNC.iloc[::n]
    df_train_sDAT=df_train_sDAT.iloc[::n]


    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

    plt.xlim(min(df_test_sDAT.iloc[:,0].min(), df_train_sDAT.iloc[:,0].min())-0.1,
                 max(df_test_sNC.iloc[:,0].max(), df_train_sNC.iloc[:,0].max())+0.1)

    plt.ylim(min(df_test_sDAT.iloc[:,1].min(), df_train_sDAT.iloc[:,1].min()-0.1),
                 max(df_test_sNC.iloc[:,1].max(), df_train_sNC.iloc[:,1].max())+0.1)

    plt.scatter(green_group.iloc[:,0], green_group.iloc[:,1], color='green', marker='.')
    plt.scatter(blue_group.iloc[:,0], blue_group.iloc[:,1], color='blue', marker='.')
    plt.scatter(df_train_sNC.iloc[:,0], df_train_sNC.iloc[:,1], color='green',marker='o', label='0/sNC train')
    plt.scatter(df_train_sDAT.iloc[:,0], df_train_sDAT.iloc[:,1], color='blue',marker='o', label='1/sDAT train
')
    plt.scatter(df_test_sNC.iloc[:,0], df_test_sNC.iloc[:,1], color='green',marker='x', label='0/sNC test')
    plt.scatter(df_test_sDAT.iloc[:,0], df_test_sDAT.iloc[:,1], color='blue',marker='x', label='1/sDAT test')


    plt.legend(loc='upper left')
    plt.show()


make_graphQ1(30)
