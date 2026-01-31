from tools import *

def HasD(x, y):
# Implementation of Hassanat Distance
    HasD=0

    for i in range(0, len(x)):
        min_xy=min(x[i], y[i])
        max_xy=max(x[i], y[i])
        if min_xy >= 0:
            HasD+= 1 - (1+min_xy)/(1+max_xy)
        else:
            abs_min=abs(min_xy)
            HasD+= 1 -(1+min_xy+abs_min)/(1+max_xy+abs_min)

    return HasD


def diagnoseDAT(Xtest, data_dir):

    path=data_dir
    if not path.endswith('/'):
        path+='/'

    df_train_sNC=pd.read_csv(path+"train.sNC.csv", header=None)
    df_train_sNC.columns=Global.columns_names
    df_train_sNC[Global.label_name]=Global.sNC_label

    df_train_sDAT=pd.read_csv(path+"train.sDAT.csv", header=None)
    df_train_sDAT.columns=Global.columns_names
    df_train_sDAT[Global.label_name]=Global.sDAT_label

    df_test_sNC=pd.read_csv(path+"test.sNC.csv", header=None)
    df_test_sNC.columns=Global.columns_names
    df_test_sNC[Global.label_name]=Global.sNC_label

    df_test_sDAT=pd.read_csv(path+"test.sDAT.csv", header=None)
    df_test_sDAT.columns=Global.columns_names
    df_test_sDAT[Global.label_name]=Global.sDAT_label

    df_train=pd.concat([df_train_sNC, df_train_sDAT], axis=0, ignore_index=True)
    df_test=pd.concat([df_test_sNC, df_test_sDAT], axis=0, ignore_index=True)

    df_all = pd.concat([df_test, df_train], axis=0, ignore_index=True)

    scaler=StandardScaler()
    X=df_all[Global.columns_names]
    X_scaled=scaler.fit_transform(X)
    Y=df_all[Global.label_name]

    print()

    k=17
    knn = KNeighborsClassifier(n_neighbors=k,
                               metric=HasD)
    knn.fit(X_scaled, Y)

    Xtest_scaled = scaler.transform(Xtest)

    ytest =  knn.predict(Xtest_scaled)

    return ytest

def Q4_test():

    # Generate a 100x2 matrix
    # low = 1.3, high = 2.2, size = (rows, columns)
    X = np.random.uniform(1, 2.2, size=(100, 2))
    data_path="Data"
    ytest=diagnoseDAT(X, data_path)

    print(ytest)
