import numpy as np
import pandas as pd
import random
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR, SVC
from sklearn_extra.cluster import KMedoids
from xgboost import XGBRegressor, XGBClassifier

from src.explain_model import *
from src.feature_selector import *
from src.finetune_model import *
from src.preprocess import *
from src.recommendation_system import *
from src.train_nn import *

RANDOM_SEED = 13


def cluster(data, cluster_option: str = "kmeans", max_num_datapoints: int = 100):
    """
    :param data: the list of datapoints
    :param cluster_option: suggests which clusterizing algorithm to try
    :param max_num_datapoints: specifies how many datapoints to use from the list data (should not be very large as it might crash matplotlib plotting capabilities
    """
    random.shuffle(data)
    if max_num_datapoints is not None:
        data = data[:max_num_datapoints]
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    if cluster_option == "kmeans":
        clusterizer = KMeans(n_clusters=5)
    elif cluster_option == "kmedoids":
        clusterizer = KMedoids()
    else:
        raise Exception("Wrong cluster_option given!")
    clusterizer.fit(data)
    labels = clusterizer.labels_
    for n_components in [2, 3]:
        for dim_red_option in ["PCA", "TSNE", "LDA", "UMAP"]:  # , "encoder"]:
            scaler = get_scaler()
            scaled_data = scaler.fit_transform(data)
            dim_reducer = load_dim_reducer(dim_red_option, n_components)
            if dim_red_option != "encoder":
                reduced_data = dim_reducer.fit_transform(scaled_data)
            elif dim_red_option == "encoder":
                reduced_data = dim_reducer.predict(scaled_data)
            else:
                raise Exceptiong("wrong dim_red_option given!")
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


def train_model(data: np.ndarray,
                labels: typing.List[float],
                task_type: str = "classification",  # typing.Literal['classification', 'regression']
                model_type: str = "SVM",
                scaling_option: str = "standard",
                use_class_weight: bool = True):  # typing.Literal['XGB', 'RF', 'NN', 'SVM']
    data_scaler = get_scaler(scaling_option=scaling_option)
    label_scaler = get_scaler(scaling_option=scaling_option)
    if use_class_weight:
        class_weight = get_class_weight(labels)
    else:
        class_weight = None
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
            model = XGBClassifier(random_state=RANDOM_SEED, class_weight=class_weight)
        elif model_type == "SVM":
            model = SVC(random_state=RANDOM_SEED, class_weight=class_weight)
        elif model_type == "RF":
            model = RandomForestClassifier(random_state=RANDOM_SEED, class_weight=class_weight)
        elif model_type == "NN":
            train_nn(X_train, y_train, X_test, y_test, model_name="nn_classifier", num_classes=len(intervals), class_weight=class_weight)
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


from scipy import stats


def main():
    df = pd.read_csv("data/beer_profile_and_ratings.csv")
    print(df.columns)
    print(df.dtypes)
    data_columns = ['Min IBU', 'Max IBU', 'Astringency', 'Body', 'Alcohol', 'Bitter', 'Sweet', 'Sour', 'Salty',
                    'Fruits', 'Hoppy', 'Spices', 'Malty']
    data = df[data_columns].to_numpy()

    abv_target = "ABV"
    review_target = "review_overall"
    plot_hist = False
    predicted_target = review_target  # abv_target

    targets = np.array(df[predicted_target].to_list())  # BREW ALCOHOL CONTENT
    print(df[predicted_target].describe())

    if plot_hist:
        plt.hist(targets, bins=25)
        plt.savefig(f"data/images/{predicted_target}_histogram.png")
        plt.title(f"{predicted_target} histogram")
        plt.show()

    taste_cols = ['Bitter', 'Sweet', 'Sour', 'Salty']
    mouthfeel_cols = ['Astringency', 'Body', 'Alcohol']
    flavor_aroma_cols = ['Fruits', 'Hoppy', 'Spices', 'Malty']

    data = df[taste_cols].to_numpy()

    cluster(data)
    # model_types = ['SVM', 'RF', 'XGB', 'NN']
    # task_types = ['regression', 'classification']
    # for model_type in model_types:
    #     for task_type in task_types:
    #         train_model(data, targets, model_type=model_type, task_type=task_type)

    # TODO-urile de mai jos sunt mai mult idei, nu trebuie sa le facem pe toate, dar sa facem cat mai multe ideal

    # TODO de vazut daca e nevoie de any data preprocessing / data cleaning / data cleansing mai speciala
    # TODO de facut analiza exploratorie in R

    # TODO de generat date pentru un system de recomandare care sa recomande pe cele 2 paradigme de recommendation ->
    # (user based si content based), ne putem inspira dintr-un kaggle notebook existent pe kaggle-ul datasetului ->
    # pe content based putem sa si recomandam berile care au o valori apropiate la feature-uri sau care produc cosine distance bun la embedding-urile ->
    # obtinute la mijlocul autoencoder-ului


if __name__ == '__main__':
    main()
