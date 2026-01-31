#  Assignment 1
#
#  Group 17:
#  <Group Member 1 name> <Group Member 1 email>
#  Greg de Souza> <gdesouza@mun.ca
#  <Group Member 3 name> <Group Member 1 email>


####################################################################################
# Imports
####################################################################################
import sys
import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score



#####################################################################################
#  Implementation - helper functions and question wise result generators
#####################################################################################
class Global:
    #Labels and Graphs according o assignment instructions
    sNC_label=0
    sNC_color='green'

    sDAT_label=1
    sDAT_color='blue'

    train_marker='o'
    test_marker='x'

    columns_names=['x1','x2']
    label_name='Target'

    #path for the training & test data provded
    sNC_train_path="Data/train.sNC.csv"
    sDAT_train_path="Data/train.sDAT.csv"

    sNC_test_path="Data/test.sNC.csv"
    sDAT_test_path="Data/test.sDAT.csv"



def HasD(x, y):
# Implementation of Hassanat Distance for Q4 as a custom metric
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


#### QUESTION 1 ####
def Q1_results():

    print("Q1_results started, it will take a while, but it will genrate pretty graphs!")

    # --- 1. Load feature datasets ---
    train_sNC_features = pd.read_csv(Global.sNC_train_path, header=None)
    train_sDAT_features = pd.read_csv(Global.sDAT_train_path, header=None)

    test_sNC_features = pd.read_csv(Global.sNC_test_path, header=None)
    test_sDAT_features = pd.read_csv(Global.sDAT_test_path, header=None)

    # Loading Grid for Background coloring
    grid_df = pd.read_csv("Data/grid_df.csv")

    # --- Name columns and label target class ---
    train_sNC_features.columns=Global.columns_names
    train_sNC_features[Global.label_name]=Global.sNC_label

    train_sDAT_features.columns=Global.columns_names
    train_sDAT_features[Global.label_name]=Global.sDAT_label

    # ---- Concatenate to Create the total training

    train_features = pd.concat([train_sNC_features, train_sDAT_features], axis=0, ignore_index=True)
    #This data set looks like: x1 | x2 | Target (y) |
    #We now creat and X matrix with x1 | x2, and the target vector y
    X_train = train_features[Global.columns_names]
    Y_train = train_features[Global.label_name]


    if X_train.empty:
        print("Error: X_train is empty! Check your CSV files for valid data.")
        return

    #Do the same thing for the test data
    test_sNC_features.columns=Global.columns_names
    test_sNC_features[Global.label_name]=Global.sNC_label

    test_sDAT_features.columns=Global.columns_names
    test_sDAT_features[Global.label_name]=Global.sDAT_label

    test_features = pd.concat([test_sNC_features, test_sDAT_features], axis=0, ignore_index=True)
    X_test = test_features[Global.columns_names]
    Y_test = test_features[Global.label_name]

    if X_test.empty:
        print("Error: X_test  is empty! Check your CSV files for valid data.")
        return


    #Create array to track errros for graphing purposes
    err_train_array=[]
    err_test_array=[]
    k_array=[]


    # --- 3. Train and plot ---
    k_values = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]


    # Plotting Stuff (Feel free to Skip)
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(32,24), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=2./72., h_pad=2./72.,
                                    hspace=0.02, wspace=0.02)
    plt.rcParams['figure.dpi'] = 100

    axes_list = axes.flatten()
    ax_index=0

    skip=2 #We plot one every <skip> points in the data setting for better vizualisation

    plot_train_nSC=train_sNC_features[::skip]
    plot_train_nDAT=train_sDAT_features[::skip]
    plot_test_nSC=test_sNC_features[::skip]
    plot_test_nDAT=test_sDAT_features[::skip]

    x1_range= (min(plot_train_nDAT['x1'].min(),
                   plot_test_nDAT['x1'].min()),
               max(plot_train_nSC['x1'].max(),
                   plot_test_nSC['x1'].max()))

    x2_range= (min(plot_train_nDAT['x2'].min(),
                   plot_test_nDAT['x2'].min()),
               max(plot_train_nSC['x2'].max(),
                   plot_test_nSC['x2'].max()))


    X_grid=grid_df[Global.columns_names]
    bg_colors=['#b2ffa9', '#889eff'] #Light Blue and Light Green
    cmap_background = ListedColormap(bg_colors)
    sqr_size=3
    alpha_value=0.3

    # End of plotting stuff (for now)

    for k in k_values:

        # Creating the Classifier
        clf = KNeighborsClassifier(k, metric='euclidean')
        #Training the classifier
        clf.fit(X_train, Y_train)

        #Getting errors
        err_train = 1-clf.score(X_train, Y_train)
        err_test = 1-clf.score(X_test, Y_test)

        #Appending the Information
        err_train_array.append(err_train)
        err_test_array.append(err_test)
        k_array.append(k)

        # Plotting Part II

        # For each position in the 3x4 Grid I'll plot a k

        ax=axes_list[ax_index]
        ax_index+=1

        ax.set_title(f"kNN Classifier for k={k} \n Training Error = {err_train:.3f} , Test Error={err_test:.3f} \n plotting 1 in every {skip} points", fontsize='large')

        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_xlim(x1_range)
        ax.set_ylim(x2_range)
        ax.set_box_aspect(1)

        grid_df[Global.label_name]=clf.predict(X_grid)

        #I Create two background coloring masks for each color/region
        grid_sNC = grid_df[grid_df[Global.label_name] == Global.sNC_label]
        grid_sDAT = grid_df[grid_df[Global.label_name] == Global.sDAT_label]

        #And plot them
        ax.scatter(grid_sNC['x1'], grid_sNC['x2'],
                    c=bg_colors[0],
                    marker='s', s=sqr_size,
                    edgecolor='none', alpha=alpha_value)
        ax.scatter(grid_sDAT['x1'], grid_sDAT['x2'],
                   c=bg_colors[1],
                   marker='s', s=sqr_size,
                   edgecolor='none', alpha=alpha_value)

        #And write their meanings on the graph itself
        regions = {
            1: {'pos': (0.35, 0.03), 'name': 'sDAT Region'},
            0: {'pos': (0.85, 0.6), 'name': 'sNC \n Region'}
        }

        for val, style in regions.items():
            ax.text(*style['pos'], style['name'], transform=ax.transAxes)


        #Now I plot the Train & Test Data
        ax.scatter(plot_train_nSC['x1'], plot_train_nSC['x2'],
                    color=Global.sNC_color,marker='o', label='0/sNC train')
        ax.scatter(plot_train_nDAT['x1'], plot_train_nDAT['x2'],
                    color=Global.sDAT_color,marker='o', label='1/sDAT train')

        ax.scatter(plot_test_nSC['x1'], plot_test_nSC['x2'],
                    color=Global.sNC_color,marker='x', label='0/sNC test')
        ax.scatter(plot_test_nDAT['x1'], plot_test_nDAT['x2'],
                    color=Global.sDAT_color,marker='x', label='1/sDAT test')


        ax.legend(loc='upper left', fontsize='large')

        # # 2. Grab the bounding box of just this subplot
        # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        # filename=f"Q1_Graphs/knn_k{k}.png"
        # # 3. Save only that extent
        # fig.savefig(filename, bbox_inches=extent.expanded(1.1, 1.1))



    # And I still have two spaces on the grid, so I'll plot some additional data
    ax=axes_list[ax_index]
    ax_index+=1

    ax.set_box_aspect(1)
    ax.set_xlabel(r'log(k)')
    ax.set_ylabel("Error")
    ax.set_xlim(1, 200)
    ax.set_xscale('log')
    ax.set_ylim(0, 0.25)
    ax.set_title("Errro vs hyperparameter K", fontsize='large')

    #The 11th plot is the Error vs k
    min_error=min(err_test_array)
    ax.plot(k_array, err_test_array, marker='x', linestyle='-', label='Testing Error')
    ax.plot(k_array, err_train_array, marker='o', linestyle='-', label='Training Error')
    ax.axhline(y=min_error, color='black', linestyle='--', label=f'Minimum Testing Error {min_error:.2f}')

    ax.legend(fontsize='large')

    ax=axes_list[ax_index]

    ax.set_box_aspect(1)
    ax.set_xlabel(r'x_1')
    ax.set_ylabel(r'x_2')
    ax.set_title("All Avaiable Data", fontsize='large')

    #our last plot is just all the data for context
    ax.scatter(train_sNC_features['x1'], train_sNC_features['x2'],
               color=Global.sNC_color,marker='o', label='0/sNC train')
    ax.scatter(train_sDAT_features['x1'], train_sDAT_features['x2'],
               color=Global.sDAT_color,marker='o', label='1/sDAT train')

    ax.scatter(test_sNC_features['x1'], test_sNC_features['x2'],
               color=Global.sNC_color,marker='x', label='0/sNC test')
    ax.scatter(test_sDAT_features['x1'], test_sDAT_features['x2'],
               color=Global.sDAT_color,marker='x', label='1/sDAT test')

    ax.legend(fontsize='large')


    #plt.tight_layout()
    #plt.show()
    plt.savefig("Q1_grid.png", dpi=300)
    print("Saving all the pretty graphs to Q1_grid.png file")
    plt.cla()
    plt.close('all')


def Q2_results():
    print("Q2 results started")
        # --- 1. Load feature datasets ---
    train_sNC_features = pd.read_csv(Global.sNC_train_path, header=None)
    train_sDAT_features = pd.read_csv(Global.sDAT_train_path, header=None)

    test_sNC_features = pd.read_csv(Global.sNC_test_path, header=None)
    test_sDAT_features = pd.read_csv(Global.sDAT_test_path, header=None)

    # Loading Grid for Background coloring
    grid_df = pd.read_csv("Data/grid_df.csv")

    # --- 2. Name columns and label target class ---
    train_sNC_features.columns=Global.columns_names
    train_sNC_features[Global.label_name]=Global.sNC_label


    # This is creating a column labeled "Target" and setting it to 0 (sNC)
    # I'm calling every label and target value from the Global enviroment so every
    # dataframe has the exact same names and everything

    train_sDAT_features.columns=Global.columns_names
    train_sDAT_features[Global.label_name]=Global.sDAT_label

    # ---- Concatenate to Create the total training

    train_features = pd.concat([train_sNC_features, train_sDAT_features], axis=0, ignore_index=True)
    #This data set looks like: x1 | x2 | Target (y) |
    #We now creat and X matrix with x1 | x2, and the target vector y
    X_train = train_features[Global.columns_names]
    Y_train = train_features[Global.label_name]


    if X_train.empty:
        print("Error: X_train is empty! Check your CSV files for valid data.")
        return

    #Do the same thing for the test data
    test_sNC_features.columns=Global.columns_names
    test_sNC_features[Global.label_name]=Global.sNC_label

    test_sDAT_features.columns=Global.columns_names
    test_sDAT_features[Global.label_name]=Global.sDAT_label

    test_features = pd.concat([test_sNC_features, test_sDAT_features], axis=0, ignore_index=True)
    X_test = test_features[Global.columns_names]
    Y_test = test_features[Global.label_name]

    if X_test.empty:
        print("Error: X_test  is empty! Check your CSV files for valid data.")
        return


    # We got k=30 as the best from question 1, so I don't think we need to redo that.
    k=30

    #Training the Euclidean classifier
    clf_euclidean = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    clf_euclidean.fit(X_train, Y_train)

    euc_train_error =1-clf_euclidean.score(X_train, Y_train)
    euc_test_error =1-clf_euclidean.score(X_test, Y_test)


    #Training The Manhattan Classifier
    clf_manhattan = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    clf_manhattan.fit(X_train, Y_train)

    man_train_error = 1-clf_manhattan.score(X_train, Y_train)
    man_test_error = 1-clf_manhattan.score(X_test, Y_test)



    #Plot 1 - The classification region for the Euclidean k=30

    skip=2 #We plot one every <skip> points in the data setting
    plot_train_nSC=train_sNC_features[::skip]
    plot_train_nDAT=train_sDAT_features[::skip]
    plot_test_nSC=test_sNC_features[::skip]
    plot_test_nDAT=test_sDAT_features[::skip]

    x1_range= (min(plot_train_nDAT['x1'].min(),
                   plot_test_nDAT['x1'].min()),
               max(plot_train_nSC['x1'].max(),
                   plot_test_nSC['x1'].max()))

    x2_range= (min(plot_train_nDAT['x2'].min(),
                   plot_test_nDAT['x2'].min()),
               max(plot_train_nSC['x2'].max(),
                   plot_test_nSC['x2'].max()))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,10), constrained_layout=True)
    axes_list=axes.flatten()
    ax_index=0

    ax=axes_list[ax_index]
    ax_index+=1

    X_grid=grid_df[Global.columns_names]
    bg_colors=['#b2ffa9', '#889eff'] #Light Blue and Light Green
    cmap_background = ListedColormap(bg_colors)
    sqr_size=5
    alpha_value=0.3

    ax.set_title(f"kNN Classifier for k={k} Euclidean Metric \n Training Error = {euc_train_error:.3f} , Test Error={euc_test_error:.3f} \n plotting 1 in every {skip} points", fontsize='large')

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_xlim(x1_range)
    ax.set_ylim(x2_range)
    ax.set_box_aspect(1)

    grid_df[Global.label_name]=clf_euclidean.predict(X_grid)
    grid_sNC = grid_df[grid_df[Global.label_name] == Global.sNC_label]
    grid_sDAT = grid_df[grid_df[Global.label_name] == Global.sDAT_label]


    ax.scatter(grid_sNC['x1'], grid_sNC['x2'],
                    c=bg_colors[0],
                    marker='s', s=sqr_size,
                    edgecolor='none', alpha=alpha_value)
    ax.scatter(grid_sDAT['x1'], grid_sDAT['x2'],
                   c=bg_colors[1],
                   marker='s', s=sqr_size,
                   edgecolor='none', alpha=alpha_value)

    ax.scatter(plot_train_nSC['x1'], plot_train_nSC['x2'],
                    color=Global.sNC_color,marker='o', label='0/sNC train')
    ax.scatter(plot_train_nDAT['x1'], plot_train_nDAT['x2'],
                    color=Global.sDAT_color,marker='o', label='1/sDAT train')

    ax.scatter(plot_test_nSC['x1'], plot_test_nSC['x2'],
                    color=Global.sNC_color,marker='x', label='0/sNC test')
    ax.scatter(plot_test_nDAT['x1'], plot_test_nDAT['x2'],
                    color=Global.sDAT_color,marker='x', label='1/sDAT test')

    regions = {
            1: {'pos': (0.35, 0.03), 'name': 'sDAT Region'},
            0: {'pos': (0.85, 0.6), 'name': 'sNC \n Region'}
        }

    for val, style in regions.items():
        ax.text(*style['pos'], style['name'], transform=ax.transAxes)

        ax.legend(loc='upper left', fontsize='large')


    # Plot two, same thing for the Manhattan Metric

    ax=axes_list[ax_index]

    X_grid=grid_df[Global.columns_names]
    bg_colors=['#b2ffa9', '#889eff'] #Light Blue and Light Green
    cmap_background = ListedColormap(bg_colors)
    sqr_size=3
    alpha_value=0.3

    ax.set_title(f"kNN Classifier for k={k} Manhattan Metric \n Training Error = {man_test_error:.3f} , Test Error={man_test_error:.3f} \n plotting 1 in every {skip} points", fontsize='large')

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_xlim(x1_range)
    ax.set_ylim(x2_range)
    ax.set_box_aspect(1)

    grid_df[Global.label_name]=clf_manhattan.predict(X_grid)
    grid_sNC = grid_df[grid_df[Global.label_name] == Global.sNC_label]
    grid_sDAT = grid_df[grid_df[Global.label_name] == Global.sDAT_label]


    ax.scatter(grid_sNC['x1'], grid_sNC['x2'],
                    c=bg_colors[0],
                    marker='s', s=sqr_size,
                    edgecolor='none', alpha=alpha_value)
    ax.scatter(grid_sDAT['x1'], grid_sDAT['x2'],
                   c=bg_colors[1],
                   marker='s', s=sqr_size,
                   edgecolor='none', alpha=alpha_value)

    ax.scatter(plot_train_nSC['x1'], plot_train_nSC['x2'],
                    color=Global.sNC_color,marker='o', label='0/sNC train')
    ax.scatter(plot_train_nDAT['x1'], plot_train_nDAT['x2'],
                    color=Global.sDAT_color,marker='o', label='1/sDAT train')

    ax.scatter(plot_test_nSC['x1'], plot_test_nSC['x2'],
                    color=Global.sNC_color,marker='x', label='0/sNC test')
    ax.scatter(plot_test_nDAT['x1'], plot_test_nDAT['x2'],
                    color=Global.sDAT_color,marker='x', label='1/sDAT test')

    regions = {
            1: {'pos': (0.35, 0.03), 'name': 'sDAT Region'},
            0: {'pos': (0.85, 0.6), 'name': 'sNC \n Region'}
        }

    for val, style in regions.items():
        ax.text(*style['pos'], style['name'], transform=ax.transAxes)

        ax.legend(loc='upper left', fontsize='large')

    #plt.show()
    plt.savefig("Q2.png", dpi=300)
    print("Saving graphs at Q2.png")
    plt.cla()
    plt.close('all')

def Q3_results():
    print("Q3 started")
    #Setting Metric based on Q2
    chosen_metric = 'euclidean'
# --- 1. Load feature datasets ---
    train_sNC_features = pd.read_csv(Global.sNC_train_path, header=None)
    train_sDAT_features = pd.read_csv(Global.sDAT_train_path, header=None)

    test_sNC_features = pd.read_csv(Global.sNC_test_path, header=None)
    test_sDAT_features = pd.read_csv(Global.sDAT_test_path, header=None)

    # --- 2. Name columns and label target class ---
    train_sNC_features.columns=Global.columns_names
    train_sNC_features[Global.label_name]=Global.sNC_label


    # This is creating a column labeled "Target" and setting it to 0 (sNC)
    # I'm calling every label and target value from the Global enviroment so every
    # dataframe has the exact same names and everything

    train_sDAT_features.columns=Global.columns_names
    train_sDAT_features[Global.label_name]=Global.sDAT_label

    # ---- Concatenate to Create the total training

    train_features = pd.concat([train_sNC_features, train_sDAT_features], axis=0, ignore_index=True)
    #This data set looks like: x1 | x2 | Target (y) |
    #We now creat and X matrix with x1 | x2, and the target vector y
    X_train = train_features[Global.columns_names]
    Y_train = train_features[Global.label_name]


    if X_train.empty:
        print("Error: X_train is empty! Check your CSV files for valid data.")
        return

    #Do the same thing for the test data
    test_sNC_features.columns=Global.columns_names
    test_sNC_features[Global.label_name]=Global.sNC_label

    test_sDAT_features.columns=Global.columns_names
    test_sDAT_features[Global.label_name]=Global.sDAT_label

    test_features = pd.concat([test_sNC_features, test_sDAT_features], axis=0, ignore_index=True)
    X_test = test_features[Global.columns_names]
    Y_test = test_features[Global.label_name]

    if X_test.empty:
        print("Error: X_test  is empty! Check your CSV files for valid data.")
        return



    k_values=range(1, 101)
    array_1overk=[]
    array_test_err=[]
    array_train_err=[]

    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k, metric=chosen_metric)
        clf.fit(X_train, Y_train)

        train_err = 1 - clf.score(X_train, Y_train)
        test_err = 1 - clf.score(X_test, Y_test)


        array_train_err.append(train_err)
        array_test_err.append(test_err)
        array_1overk.append(float(1/k))

    min_err = min(array_test_err)

    plt.xlabel(r'Model Capacity ($log(\frac{1}{k})$)')
    plt.ylabel(r'Error on Log Scale')

    plt.plot(array_1overk, array_test_err, marker='None',
             linestyle='--', label='Testing Error', color='cyan')
    plt.plot(array_1overk, array_train_err, marker='None', linestyle='-',
             label='Training Error', color='orange')
    plt.axhline(y=min_err, color='black', linestyle=':',
                label=f'Minimum Testing Error: {min_err:.4f}')

    plt.xscale('log')
    plt.title("Error vs Model Capacity for kNN with Euclidean Metric")
    plt.grid(True, which="both", ls="-", alpha=0.3)

    plt.legend(loc='lower left')
    plt.savefig('Q3.png', dpi=300)
    print("Q3 graph saved as Q3.png")
    plt.close('all')


def diagnoseDAT(Xtest, data_dir):

    #Check if the data path has / at the end, otherwise add it
    path=data_dir
    if not path.endswith('/'):
        path+='/'

    #load ALL the data I can
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

    #And throw it all in a single DF
    df_all = pd.concat([df_test, df_train], axis=0, ignore_index=True)

    #Strategy 1: Scaling the Data
    scaler=StandardScaler()
    X=df_all[Global.columns_names]
    X_scaled=scaler.fit_transform(X)
    Y=df_all[Global.label_name]

    #Our exploration suggested that k=17 for the Hassamat Metric gives good results
    k=17
    knn = KNeighborsClassifier(n_neighbors=k,
                               metric=HasD)
    knn.fit(X_scaled, Y)

    #Scale the input data
    Xtest_scaled = scaler.transform(Xtest)

    #Generate the prediction
    ytest =  knn.predict(Xtest_scaled)

    return ytest



#########################################################################################
# Calls to generate the results
#########################################################################################

if __name__=="__main__":
    Q1_results()
    Q2_results()
    Q3_results()

    try:
        print("Starting diagnoseDat(Xtest, data_dir)")
        ytest=diagnoseDAT(Xtest, data_dir)
    except:
        print("Exception: diagnoseDat arguments not well defined")
