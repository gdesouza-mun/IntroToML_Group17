import pandas as pd
import matplotlib.pyplot as plt


def remake_grid():
    df=pd.read_csv('Grids/2D_grid_points.csv', header=None)
    df['x>y'] = df.iloc[:,0]>=df.iloc[:,1]

    blue_group=df[df['x>y']==True]
    green_group=df[df['x>y']==False]

    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

    plt.scatter(green_group.iloc[:,0], green_group.iloc[:,1], color='green')
    plt.scatter(blue_group.iloc[:,0], blue_group.iloc[:,1], color='blue')

    plt.show()

def train_scatter():
    df_sNC=pd.read_csv('Data/train.sNC.csv', header=None)
    df_sDAT=pd.read_csv('Data/train.sDAT.csv', header=None)

    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.scatter(df_sNC.iloc[:,0], df_sNC.iloc[:,1], color='green',marker='o', label='0/sNC')
    plt.scatter(df_sDAT.iloc[:,0], df_sDAT.iloc[:,1], color='blue',marker='o', label='1/sDAT')

    plt.legend()
    plt.title("Assig 1 Training Data")

    plt.show()

def all_data(every=1):

    df_test_sNC=pd.read_csv('Data/test.sNC.csv', header=None)
    df_test_sDAT=pd.read_csv('Data/test.sDAT.csv', header=None)
    df_train_sNC=pd.read_csv('Data/train.sNC.csv', header=None)
    df_train_sDAT=pd.read_csv('Data/train.sDAT.csv', header=None)

    df_test_sNC=df_test_sNC.iloc[::every]
    df_test_sDAT=df_test_sDAT.iloc[::every]
    df_train_sNC=df_train_sNC.iloc[::every]
    df_train_sDAT=df_train_sDAT.iloc[::every]


    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

    plt.xlim(min(df_test_sDAT.iloc[:,0].min(), df_train_sDAT.iloc[:,0].min())-0.1,
                 max(df_test_sNC.iloc[:,0].max(), df_train_sNC.iloc[:,0].max())+0.1)

    plt.ylim(min(df_test_sDAT.iloc[:,1].min(), df_train_sDAT.iloc[:,1].min()-0.1),
                 max(df_test_sNC.iloc[:,1].max(), df_train_sNC.iloc[:,1].max())+0.1)

    plt.scatter(df_train_sNC.iloc[:,0], df_train_sNC.iloc[:,1],
                color='green',marker='o', label='0/sNC train')
    plt.scatter(df_train_sDAT.iloc[:,0], df_train_sDAT.iloc[:,1],
                color='blue',marker='o', label='1/sDAT train')

    plt.scatter(df_test_sNC.iloc[:,0], df_test_sNC.iloc[:,1],
                color='green',marker='x', label='0/sNC test')
    plt.scatter(df_test_sDAT.iloc[:,0], df_test_sDAT.iloc[:,1],
                color='blue',marker='x', label='1/sDAT test')

    plt.title("Assigment 1 Data ploted every "+str(every)+" points")
    plt.legend(loc='upper left')
    plt.show()

all_data(2)
