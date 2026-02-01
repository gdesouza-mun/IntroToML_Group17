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


def plot_decision_region(knn, scaler, x1_range=None, x2_range=None, step=0.02):
# 1. Create a range of values for X and Y using pure Python/Pandas
    # We create a list from min to max with 'step' increments
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

    grid_scaled=scaler.transform(grid_df)

    # 3. Predict the classes for the entire grid
    # grid_df now looks exactly like your training features

    grid_df['label'] = knn.predict(grid_scaled)

    # 4. Plotting using a Scatter Plot as a "Heatmap"
    # Since we can't use contourf (which requires 2D arrays),

    # we use a dense scatter plot as the background.
    bg_colors=['#b2ffa9', '#889eff']
    cmap_background = ListedColormap(bg_colors) # Light Red, Light Blue
    labels = ['Class 0: sNC', 'Class 1: sDAT']

    legend_labels = [mpatches.Patch(color=bg_colors[i], label=labels[i], alpha=0.5)
               for i in range(len(bg_colors))]


    plt.scatter(grid_df['x1'], grid_df['x2'],
                c=grid_df['label'],
                cmap=cmap_background,
                marker='s', s=3, alpha=0.5) # 's' is a square marker


def diagnoseDAT(path):

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

    x1_range= (df_all['x1'].min(), df_all['x1'].max())
    x2_range= (df_all['x2'].min(), df_all['x2'].max())


    plt.scatter(df_train_sNC.iloc[:,0], df_train_sNC.iloc[:,1],
                        color=Global.sNC_color,marker='o', label='0/sNC train')
    plt.scatter(df_train_sDAT.iloc[:,0], df_train_sDAT.iloc[:,1],
                        color=Global.sDAT_color,marker='o', label='1/sDAT train')

    plt.scatter(df_test_sNC.iloc[:,0], df_test_sNC.iloc[:,1],
                        color=Global.sNC_color,marker='x', label='0/sNC test')
    plt.scatter(df_test_sDAT.iloc[:,0], df_test_sDAT.iloc[:,1],
                        color=Global.sDAT_color,marker='x', label='1/sDAT test')


    plot_decision_region(knn, scaler, x1_range, x2_range, step=0.002)

    plt.title("Decision Boundary for kNN with Hassanat Metric and k=17")
    plt.legend()

    plt.savefig("Q4.png", dpi=300)

path=("Data/")
diagnoseDAT(path)
