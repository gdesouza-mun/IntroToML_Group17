import pandas as pd
import matplotlib.pyplot as plt
import math
import statistics

class glb:
    feat_dic = [
        "Cement_component1__kgInAM_3Mixture_",
        "BlastFurnaceSlag_component2__kgInAM_3Mixture_",
        "FlyAsh_component3__kgInAM_3Mixture_",
        "Water_component4__kgInAM_3Mixture_",
        "Superplasticizer_component5__kgInAM_3Mixture_",
         "CoarseAggregate_component6__kgInAM_3Mixture_",
         "FineAggregate_component7__kgInAM_3Mixture_",
         "Age_day_"
    ]

    target_name = "ConcreteCompressiveStrength_MPa_Megapascals_"

    p_features=8

    train_Filepath="Data/train.csv"
    test_Filepath="Data/test.csv"

    def MSEtoRSE(MSE, sample_size, p=p_features):
        RSE = math.sqrt((sample_size*MSE)/(sample_size-p-1))

        return RSE



def dataVisu():
    df_train = pd.read_csv('Data/train.csv')
    df_test = pd.read_csv('Data/test.csv')

    print(df_train[glb.feat_dic[0]])

    y=df_train[glb.target_name]

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16,32))

    axes_list=axes.flatten()

    for i_feat in range(len(glb.feat_dic)):
        ax=axes_list[i_feat]
        ax.plot(df_train[glb.feat_dic[i_feat]], y, linestyle='', marker='.')

        ax.set_xlabel(f"x_{i_feat:.0f}")
        ax.set_ylabel("y")
        ax.set_title(f"x_{i_feat:.0f} vs y")

    plt.show()


def dataVisu2():
    df_train = pd.read_csv('Data/train.csv')
    df_test = pd.read_csv('Data/test.csv')

    y=df_train[glb.target_name]
    print(statistics.mean(y))
    # X = df_train[glb.feat_dic[4]]

    # for i in range(len(glb.feat_dic[7])):
    #     X[i] = math.log(X[i])

    # plt.plot(df_train[glb.feat_dic[7]], y, linestyle='', marker='.')
    # plt.show()



#dataVisu2()

#y mean=35.57
