import numpy as np
import pandas as pd
import typing
import umap
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_dim
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
# TODO research box cox
from xgboost import XGBRegressor, XGBClassifier

from src.nn import train_nn

from sklearn_extra.cluster import KMedoids
def cluster(data, cluster_option: str = "kmedoids"):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    if cluster_option == "kmeans":
        clusterizer = KMeans()
    elif cluster_option == "kmedoids":
        clusterizer = KMediods()
    else:
        raise Exception("Wrong cluster_option given!")
    clusterizer.fit(data)
    labels = clusterizer.labels_
    for n_components in [2, 3]:
        for dim_red_option in ["PCA", "TSNE", "LDA", "UMAP"]:
            if dim_red_option == "PCA":
                dim_reducer = PCA(n_components=n_components)
            elif dim_red_option == "TSNE":
                dim_reducer = TSNE(n_components=n_components)
            elif dim_red_option == "LDA":
                dim_reducer = LDA_dim(n_components=n_components)
            elif dim_red_option == "UMAP":
                dim_reducer = umap.UMAP(n_components=n_components)
            else:
                raise Exception("wrong dimensionality reduction option given!")
            scaled_data = StandardScaler().fit_transform(data)
            reduced_data = dim_reducer.fit_transform(scaled_data)
            colors = {0: "b", 1: "r", 2: "g", 3: "c", 4: "m", 5: "y", 6: "k"}
            if n_components == 2:
                for (x, label) in list(zip(reduced_data, labels)):
                    plt.scatter(x[0], x[1], color=colors[label])
                plt.gca().set_aspect('equal', 'datalim')
            elif n_components == 3:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                for (x, label) in list(zip(reduced_data, labels)):
                    ax.scatter(x[0], x[1], x[2], color=colors[label])
            else:
                raise Exception(f"n_components has to be 2 or 3 while {n_components} was given!")
            plt.title(f'{dim_red_option} learned embeddings projection of the protein agg. dataset in {n_components}D', fontsize=24)
            plt.show()


def train(data: np.ndarray,
          labels: typing.List[float],
          task_type: str = "classification",  # typing.Literal['classification', 'regression']
          model_type: str = "SVM"):  # typing.Literal['XGB', 'RF', 'NN', 'SVM']
    data_scaler = StandardScaler()
    label_scaler = StandardScaler()

    # TODO subject to change in the future
    intervals = {(min(labels), 3.999): 0,
                 (4, 4.9999): 1,
                 (5, 7.4999): 2,
                 (7.5, max(labels)): 3}

    if task_type == "classification":
        labels = np.array([intervals[(l, r)] for label in labels for (l, r) in intervals.keys() if l <= label <= r])

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    X_train = data_scaler.fit_transform(X_train)
    X_test = data_scaler.transform(X_test)

    if task_type == "regression":
        y_train = label_scaler.fit_transform(y_train.reshape(-1, 1))
        y_test = label_scaler.transform(y_test.reshape(-1, 1))

    if task_type == "classification":
        if model_type == "XGB":
            model = XGBClassifier()
        elif model_type == "SVM":
            model = SVC()
        elif model_type == "RF":
            model = RandomForestClassifier()
        elif model_type == "NN":
            train_nn(X_train, y_train, X_test, y_test, model_name="nn_classifier", num_classes=len(intervals))
        else:
            raise Exception("Wrong model_type was given")
    elif task_type == "regression":
        if model_type == "XGB":
            model = XGBRegressor()
        elif model_type == "SVM":
            model = SVR()
        elif model_type == "RF":
            model = RandomForestRegressor()
        elif model_type == "NN":
            train_nn(X_train, y_train, X_test, y_test, model_name="nn_regressor", num_classes=1)
        else:
            raise Exception("Wrong model_type was given")
    else:
        raise Exception("Wrong task_type given!")

    if model_type != "NN":
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if task_type == "classification":
            print("F1: ", f1_score(y_pred, y_test, average='weighted'))
        else:
            print("MAE:", mean_absolute_error(y_pred, y_test))


def main():
    df = pd.read_csv("data/beer_profile_and_ratings.csv")
    print(df.columns)
    print(df.dtypes)
    data_columns = ['Min IBU', 'Max IBU', 'Astringency', 'Body', 'Alcohol', 'Bitter', 'Sweet', 'Sour', 'Salty',
                    'Fruits', 'Hoppy', 'Spices', 'Malty']
    data = df[data_columns].to_numpy()
    targets = np.array(df['ABV'].to_list())  # BREW ALCOHOL CONTENT
    print(df['ABV'].describe())

    # plt.hist(targets, bins=25)
    # plt.show()
    data = df[['Bitter', 'Sweet', 'Sour', 'Salty']].to_numpy()
    cluster(data)
    # model_types = ['SVM', 'RF', 'XGB', 'NN']
    # task_types = ['regression', 'classification']
    # for model_type in model_types:
    #     for task_type in task_types:
    #         train(data, targets, model_type=model_type, task_type=task_type)


if __name__ == '__main__':
    main()
