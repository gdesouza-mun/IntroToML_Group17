import pandas as pd
import matplotlib.pyplot as plt
from dataVisu import glb
import math

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

def transform(X):
    X_new=X
    col0 = X_new[glb.feat_dic[0]].astype(float)
    col2 = X_new[glb.feat_dic[0]].astype(float)
    Age = X_new[glb.feat_dic[7]].astype(float)
    col4 = X_new[glb.feat_dic[4]].astype(float)
    col5 = X_new[glb.feat_dic[5]].astype(float)
    col6 = X_new[glb.feat_dic[6]].astype(float)
    col1_2 = X_new[glb.feat_dic[1]].astype(float)
    col_456 = []
    col_log0 = []
    col_2sqrt = []
    drop_index=[0,1,4,5,6,7]
    for i in drop_index:
        X_new=X_new.drop(glb.feat_dic[i], axis=1)


    for i in range(len(Age)):
        x0=col0[i]
        x2=col2[i]
        x4 = col4[i]
        x5 = col5[i]
        x6 = col6[i]
        x7=Age[i]
        Age[i] = math.log(x7)
        col_456.append(math.pow(x4*x5*x6, 0.5))
        col4[i] = math.pow(x2,0.5)
        col1_2[i] = math.pow(col1_2[i], 2)
        col_log0.append(math.log(x0))

    X_new["log(Age_day_)"] = Age
    X_new["Superplasticizer_component5__kgInAM_3Mixture_**1.66"] = col4
    X_new["BlastFurnaceSlag_component2__kgInAM_3Mixture_**2"] = col1_2
    X_new["x4*x5*x6"] = col_456
    X_new["logx0"] = col_log0

    return X_new


def predictCompressiveStrength(Xtest, data_dir):

    #Renaming the columns to avoid conflict in case Xtest is a pure array
    Xdf=pd.DataFrame(Xtest)
    Xdf.columns = glb.feat_dic

    path=data_dir
    if not path.endswith('/'):
        path+='/'

    alpha=0.0099
    #df1 = pd.read_csv(path+'train.csv')
    #df2 = pd.read_csv(path+'test.csv')
    df_train = pd.read_csv(path+'train.csv')
    #df_train = pd.concat([df1, df2], axis=0, ignore_index=True)

    Y=df_train[glb.target_name]
    X=df_train.drop(glb.target_name, axis=1)

    X_new = transform(X)

    model = HistGradientBoostingRegressor(
        learning_rate=0.1,
        max_iter=500,  # Sufficient trees to converge
        max_depth=None,  # Allow trees to grow (regularized by min_samples_leaf)
        min_samples_leaf=20,  # Prevents overfitting to noise
        l2_regularization=0.1,  # Slight regularization
        random_state=42
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    pipeline.fit(X_new, Y)
    Xtest_new = transform(Xdf)
    Y_out = pipeline.predict(Xtest_new)

    return Y_out




df_test = pd.read_csv('Data/test.csv')
Y=df_test[glb.target_name]
Xtest=(df_test.drop(glb.target_name, axis=1)).values

Ypred = predictCompressiveStrength(Xtest, 'Data')

r2 = r2_score(Y, Ypred)

print(r2)


# ytest(Xtest)

