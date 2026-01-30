import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tools import *

def create_grid(step, x1_range, x2_range):

    #Set x1 and x2 grid steps
    x1_values = []
    curr = x1_range[0]
    while curr <= x1_range[1]:
        x1_values.append(curr)
        curr += step

    x2_values = []
    curr = x2_range[0]
    while curr <= x2_range[1]:
        x2_values.append(curr)
        curr += step


    # 2. Build the Grid DataFrame using a Cross Join
    df_x1 = pd.DataFrame({'x1': x1_values})
    df_x2 = pd.DataFrame({'x2': x2_values})

    # This creates a row for every possible combination of x1 and x2
    grid_df = df_x1.merge(df_x2, how='cross')

    return grid_df

def Q1_results():

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

    # --- 3. Train and plot ---
    k_values = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]


    # Graphing Stuff
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15,30), constrained_layout=True)
    axes_list = axes.flatten()
    ax_index=0

    skip=3 #We plot one every <skip> points in the data setting

    plot_train_nSC=train_sNC_features[::skip]
    plot_train_nDAT=train_sDAT_features[::skip]
    plot_test_nSC=test_sNC_features[::skip]
    plot_test_nDAT=test_sDAT_features[::skip]

    x1_range= (min(plot_train_nDAT['x1'].min(),
                   plot_test_nDAT['x1'].min()) - 0.1,
               max(plot_train_nDAT['x1'].max(),
                   plot_test_nDAT['x1'].max())+0.1)

    x2_range= (min(plot_train_nDAT['x2'].min(),
                   plot_test_nDAT['x2'].min()) - 0.1,
               max(plot_train_nDAT['x2'].max(),
                   plot_test_nDAT['x2'].max())+0.1)

    grid_density=0.002
    sqr_size=3
    alpha_value=0.3
    grid_df = create_grid(grid_density, x1_range, x2_range)
    X_grid=grid_df[Global.columns_names]
    bg_colors=['#b2ffa9', '#889eff'] #Light Blue and Light Green
    cmap_background = ListedColormap(bg_colors)


    for k in k_values:

        # Creating the Classifier
        clf = KNeighborsClassifier(k, metric='euclidean')
        clf.fit(X_train, Y_train)

        #Getting errors
        err_train = 1-clf.score(X_train, Y_train)
        err_test = 1-clf.score(X_test, Y_test)

        #print(f"{k}\t{err_train}\t{err_test}")


        # 4. ------ Graphing -------


        ax=axes_list[ax_index]
        ax_index+=1

        ax.set_title(f"kNN Classifier for k={k} \n Training Error = {err_train:.3f} , Test Error={err_test:.3f}")

        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_xlim(x1_range)
        ax.set_ylim(x2_range)
        ax.set_box_aspect(1)

        grid_df[Global.label_name]=clf.predict(X_grid)

        grid_sNC = grid_df[grid_df[Global.label_name] == Global.sNC_label]
        grid_sDAT = grid_df[grid_df[Global.label_name] == Global.sDAT_label]

        # ax.scatter(grid_sNC['x1'], grid_sNC['x2'],
        #             c=bg_colors[0], label="sNC Region",
        #             marker='s', s=sqr_size,
        #             edgecolor='none', alpha=alpha_value)
        # ax.scatter(grid_sDAT['x1'], grid_sDAT['x2'],
        #            c=bg_colors[1], label="sDAT Region",
        #            marker='s', s=sqr_size,
        #            edgecolor='none', alpha=alpha_value)

        ax.scatter(plot_train_nSC['x1'], plot_train_nSC['x2'],
                    color=Global.sNC_color,marker='o', label='0/sNC train')
        ax.scatter(plot_train_nDAT['x1'], plot_train_nDAT['x2'],
                    color=Global.sDAT_color,marker='o', label='0/sDAT train')

        ax.scatter(plot_test_nSC['x1'], plot_test_nSC['x2'],
                    color=Global.sNC_color,marker='o', label='0/sNC test')
        ax.scatter(plot_test_nDAT['x1'], plot_test_nDAT['x2'],
                    color=Global.sDAT_color,marker='o', label='0/sDAT test')

        ax.legend(loc='upper left')

        # # 2. Grab the bounding box of just this subplot
        # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        # filename=f"Q1_Graphs/knn_k{k}.png"
        # # 3. Save only that extent
        # fig.savefig(filename, bbox_inches=extent.expanded(1.1, 1.1))


    #plt.tight_layout()
    plt.savefig("Q1_grid.png", dpi=300)
    #plt.show()


    # Plot in 3x4 grid leaving 2 empty
