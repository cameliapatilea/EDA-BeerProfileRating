import numpy as np
import pandas as pd
import random
import typing
# import umap.umap_ as umap
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_dim
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR, SVC
from sklearn_extra.cluster import KMedoids
from xgboost import XGBRegressor, XGBClassifier

# from src.explain_model import *
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
        clusterizer = KMeans(n_clusters=4)
    elif cluster_option == "kmedoids":
        clusterizer = KMedoids()
    else:
        raise Exception("Wrong cluster_option given!")
    clusterizer.fit(data)
    labels = clusterizer.labels_
    for n_components in [2, 3]:
        for dim_red_option in ["FA", "PCA", "TSNE", "SVD"]:  # , "UMAP", "LDA"]:  # , "encoder"]:
            scaler = get_scaler()
            scaled_data = scaler.fit_transform(data)
            dim_reducer = load_dim_reducer(dim_red_option, n_components)
            if dim_red_option != "encoder":
                reduced_data = dim_reducer.fit_transform(scaled_data)
            elif dim_red_option == "encoder":
                reduced_data = dim_reducer.predict(scaled_data)
            else:
                raise Exception("wrong dim_red_option given!")
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
                use_class_weight: bool = False):  # typing.Literal['XGB', 'RF', 'NN', 'SVM']
    data_scaler = get_scaler(scaling_option=scaling_option)
    label_scaler = get_scaler(scaling_option=scaling_option)
    if use_class_weight:
        class_weight = get_class_weight(labels)
        print(class_weight)
    else:
        class_weight = None
    # TODO subject to change in the future
    intervals = {(min(labels), 3.999): 0,
                 (4, 4.9999): 1,
                 (5, 7.4999): 2,
                 (7.5, max(labels)): 3}
    # intervals =     {(0, 2.999): 0,
    #                 (3, 3.4999): 1,
    #                 (3.5, 3.7499): 2,
    #                 (3.75, 3.9999): 3,
    #                 (4, 5): 4}
    
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
            train_nn(X_train, y_train, X_test, y_test, model_name="nn_classifier", num_classes=len(labels), class_weight=class_weight)
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
        # feature_selector(data, labels, model_type, data_columns, len(data_columns), option=feature_selection_option)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if task_type == "classification":
            print(classification_report(y_test, y_pred))
            plot_heatmap(y_pred,y_test)
        else:
            print(model_type + " MAE:", mean_absolute_error(y_pred, y_test))


from scipy import stats


def main():
    df = pd.read_csv("eda_dataset/beer_profile_and_ratings.csv")
    print(df.columns)
    print(df.dtypes)
    print(df.isna().sum())
    data_columns = ["ABV", 'Min IBU', 'Max IBU', 'Astringency', 'Body', 'Alcohol', 'Bitter', 'Sweet', 'Sour', 'Salty',
                    'Fruits', 'Hoppy', 'Spices', 'Malty', 'review_overall']
    
    abv_target = "ABV"  # regression
    review_target = "review_overall" #classification

    data_columns.remove(abv_target)
    # data_columns.remove(review_target)

    data = df[data_columns].to_numpy()

    
    alcohol_target = "Alcohol" # classification

    task_types = ['regression', 'classification']
    # if option in task_types:
    #     task_types = [option]
    #     if option == "regression":
    #         predicted_target = abv_target
    #     elif option == "classification":
    #         predicted_target = abv_target
    # else:
    #     task_types = [option]

    if "cluster" in task_types:
        taste_cols = ['Bitter', 'Sweet', 'Sour', 'Salty']
        # mouthfeel_cols = ['Astringency', 'Body', 'Alcohol']
        flavor_aroma_cols = ['Fruits', 'Hoppy', 'Spices', 'Malty']
        for types_of_cols in [taste_cols, flavor_aroma_cols]:
            data = df[types_of_cols].to_numpy()
            print(types_of_cols, "**" * 68)
            # cluster(data)

    

    plot_hist = False
    targets_regression = np.array(df[abv_target].to_list())  # BREW ALCOHOL CONTENT
    targets_classification = np.array(df[abv_target].to_list())

    print(len(targets_regression))
    print(len(targets_classification))

    print(df[abv_target].describe())

    if plot_hist:
        plt.hist(targets_regression, bins=100)
        # plt.savefig(f"data/images/{predicted_target}_histogram.png")
        plt.title(f"{abv_target} histogram")
        plt.show()

    feature_selection_options_list = ["VarianceThreshold", "SelectKBest"]#, "RFE", "SelectFromModel"] # the last 2 have some bug

    regression_models = [SVR(), RandomForestRegressor(), XGBRegressor()]
    classification_models = [SVC(), RandomForestClassifier(), XGBClassifier()]

    # intervals =     {(0, 3): 0,
    #                 (3, 3.4999): 1,
    #                 (3.5, 3.7499): 2,
    #                 (3.75, 3.9999): 3,
    #                 (4, max(targets_regression)): 4}
    # targets_classification = np.array([intervals[(l, r)] for label in targets_regression for (l, r) in intervals.keys() if l <= label <= r])


    # for task_type in task_types:
    #     if task_type == "classification":
    #         used_models = classification_models
    #     elif task_type == "regression":
    #         used_models = regression_models
    #     else:
    #         raise Exception("wrong task type!")
    #     for used_model in used_models:
    #         for feature_selection_option in feature_selection_options_list:
    #             print(task_type, used_model)
    #             if task_type == "classification":
    #                 feature_selector(data, labels, used_model, data_columns, len(data_columns), option=feature_selection_option)
    #             if task_type == "regression":
    #                 feature_selector(data, targets, used_model, data_columns, len(data_columns), option=feature_selection_option)

    # # exit(0)
    
    # if "regression" in task_types:
    #     distributions = {'gamma': [0.001, 0.1],
    #                         'C': [0.1, 1, 10]}
    #     print('SVR:',finetune_model(SVR(), data, targets, distributions=distributions))

    #     distributions = {'max_depth': [10, 100, None],
    #                     'min_samples_leaf': [1, 2, 4]}
    #     print('RandomForestRegressor:',finetune_model(RandomForestRegressor(), data, targets, distributions=distributions))
    
    # if "classification" in task_types:
        
    #     distributions = {"gamma": [0.001, 0.1], 
    #                         "C": [0.1, 1, 10]}
    #     print('SVC:',finetune_model(SVC(), data, labels, distributions=distributions))

    #     distributions = {"max_depth": [10, 100, None],
    #                         "min_samples_leaf": [1, 2, 4]}
    #     print('RandomForestClassifier:',finetune_model(RandomForestClassifier(), data, labels, distributions=distributions))

    
    # exit(0)
    print(len(data), len(targets_regression), len(targets_classification))
    model_types = ['SVM', 'RF', 'XGB', 'NN']
    for model_type in model_types:
        train_model(data, targets_regression, model_type=model_type, task_type='regression')
        train_model(data, targets_classification, model_type=model_type, task_type='classification')


if __name__ == '__main__':
    main()
