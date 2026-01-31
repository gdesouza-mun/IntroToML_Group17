from tools import *


def Q3_results():
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
    plt.savefig('Q3_graph.png', dpi=300)
    #plt.show()
