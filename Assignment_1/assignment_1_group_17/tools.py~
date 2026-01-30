import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


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

def train_kNN(df, k=3, metr='euclidean', wei='uniform'):
    #Initializing the kNN instance
    knn=KNeighborsClassifier(k, metric=str(metr), weights=str(wei))
    #And Scaled
    scaler=StandardScaler()

    #Joining the 0 and 1 labeled training set
    X=df[Global.columns_names]
    X_scaled=scaler.fit_transform(X)
    Y=df[Global.label_name]

    knn.fit(X_scaled, Y)

    err_train=1-knn.score(X_scaled, Y)

    return knn, scaler, err_train

def test_kNN(df, knn, scaler):
    X=df[Global.columns_names]
    X_scaled=scaler.transform(X)
    Y=df[Global.label_name]

    err_test=1-knn.score(X_scaled, Y)

    return err_test

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
                marker='s', s=10, alpha=0.5) # 's' is a square marker

    #plt.show()
