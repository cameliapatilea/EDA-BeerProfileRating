import numpy as np
import pandas as pd
import typing
from sklearn.feature_selection import RFE, VarianceThreshold, SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_regression  # ,
from sklearn.preprocessing import PolynomialFeatures


def add_features_via_polynomial(data, degree):
    return PolynomialFeatures(degree).fit_transform(data)


def feature_selector(X_train, y_train, model, original_features: typing.List[str], num_features=10, option="RFE"):
    feature_selection_options_list = ["RFE", "VarianceThreshold",
                                      "SelectKBest", "SelectFromModel"]

    X_train = pd.DataFrame(data=X_train, index=list(range(len(X_train))),
                           columns=["col_" + str(i) for i in range(len(
                               X_train[0]))])

    if option == "RFE":
        selector = RFE(model, n_features_to_select=1,
                       step=1)
    elif option == "VarianceThreshold":
        selector = VarianceThreshold()
    elif option == "SelectKBest":
        selector = SelectKBest(mutual_info_regression, k=X_train.shape[1] // 1)
    elif option == "SelectFromModel":
        selector = SelectFromModel(estimator=model)
    else:
        print("No feature selection method was given!")
        return X_train

    selector.fit(X_train, y_train)
    columns_list = X_train.columns[selector.get_support()].to_numpy()

    X_train_ = []
    for i in range(len(X_train)):
        datapoint = []
        for j in range(len(columns_list)):
            datapoint.append(X_train.iloc[i][columns_list[j]])
        X_train_.append(np.array(datapoint))

    score_original_tup_list = []
    if option == "RFE":
        score_original_tup_list = list(zip(selector.ranking_, original_features))
    elif option == "VarianceThreshold":
        score_original_tup_list = list(
            zip(selector.variances_, original_features))
    elif option == "SelectKBest":
        score_original_tup_list = list(
            zip(selector.scores_, original_features))

    if len(score_original_tup_list):
        if option != 'RFE':
            score_original_tup_list = sorted(score_original_tup_list, key=lambda
                x: (-1) * x[0])
        else:
            score_original_tup_list = sorted(score_original_tup_list, key=lambda
                x: x[0])

    print(option, score_original_tup_list)
    return np.array(X_train_)
