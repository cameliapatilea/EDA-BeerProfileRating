import numpy as np
import pandas as pd
import typing
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC
# TODO research box cox
from xgboost import XGBRegressor, XGBClassifier

from src.nn import train_nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def cluster(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    clusterizer = KMeans()
    clusterizer.fit(data)
    labels = clusterizer.labels_
    dim_reducer = PCA( n_components=2)
    red_data = dim_reducer.fit_transform(data)

    print(red_data.shape)
    for (x,y) in red_data:
        plt.scatter(x,y)
    plt.show()


def train(data: np.ndarray,
          labels: typing.List[float],
          task_type: str = "classification",  # typing.Literal['classification', 'regression']
          model_type: str = "SVM"):  # typing.Literal['XGB', 'RF', 'NN', 'SVM']
    data_scaler = StandardScaler()
    label_scaler = StandardScaler()

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
