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

    # Loading Grid for Background coloring
    grid_df = pd.read_csv("grid_df.csv")

    # grid_density=0.002
    # grid_df = create_grid(grid_density, x1_range, x2_range)
    # grid_df.to_csv('grid_df.csv', index=False)

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


    err_train_array=[]
    err_test_array=[]
    k_array=[]


    # --- 3. Train and plot ---
    k_values = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]


    # Graphing Stuff
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(32,24), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=2./72., h_pad=2./72.,
                                    hspace=0.02, wspace=0.02)
    plt.rcParams['figure.dpi'] = 100

    axes_list = axes.flatten()
    ax_index=0

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


    X_grid=grid_df[Global.columns_names]
    bg_colors=['#b2ffa9', '#889eff'] #Light Blue and Light Green
    cmap_background = ListedColormap(bg_colors)
    sqr_size=3
    alpha_value=0.3

    for k in k_values:

        # Creating the Classifier
        clf = KNeighborsClassifier(k, metric='euclidean')
        clf.fit(X_train, Y_train)

        #Getting errors
        err_train = 1-clf.score(X_train, Y_train)
        err_test = 1-clf.score(X_test, Y_test)

        err_train_array.append(err_train)
        err_test_array.append(err_test)
        k_array.append(k)

        #print(f"{k}\t{err_train}\t{err_test}")


        # 4. ------ Graphing -------


        ax=axes_list[ax_index]
        ax_index+=1

        ax.set_title(f"kNN Classifier for k={k} \n Training Error = {err_train:.3f} , Test Error={err_test:.3f} \n plotting 1 in every {skip} points", fontsize='large')

        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.set_xlim(x1_range)
        ax.set_ylim(x2_range)
        ax.set_box_aspect(1)

        grid_df[Global.label_name]=clf.predict(X_grid)

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

        # # 2. Grab the bounding box of just this subplot
        # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

        # filename=f"Q1_Graphs/knn_k{k}.png"
        # # 3. Save only that extent
        # fig.savefig(filename, bbox_inches=extent.expanded(1.1, 1.1))


    ax=axes_list[ax_index]
    ax_index+=1

    ax.set_box_aspect(1)
    ax.set_xlabel(r'log(k)')
    ax.set_ylabel("Error")
    ax.set_xlim(1, 200)
    ax.set_xscale('log')
    ax.set_ylim(0, 0.25)
    ax.set_title("Errro vs hyperparameter K", fontsize='large')

    min_error=min(err_test_array)
    #print(err_test_array)
    ax.plot(k_array, err_test_array, marker='x', linestyle='-', label='Testing Error')
    ax.plot(k_array, err_train_array, marker='o', linestyle='-', label='Training Error')
    ax.axhline(y=min_error, color='black', linestyle='--', label=f'Minimum Testing Error {min_error:.2f}')

    ax.legend(fontsize='large')

    ax=axes_list[ax_index]

    ax.set_box_aspect(1)
    ax.set_xlabel(r'x_1')
    ax.set_ylabel(r'x_2')
    ax.set_title("All Avaiable Data", fontsize='large')

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
