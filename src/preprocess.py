from src.train_nn import load_from_checkpoint
# import umap.umap_ as umap
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_dim
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def box_cox_transformation(raw_data):
    # TODO test and see if this can be applied for this case
    fitted_data, fitted_lambda = stats.boxcox(raw_data)

    fig, ax = plt.subplots(1, 2)

    sns.distplot(raw_data, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 2},
                 label="Non-Normal", color="green", ax=ax[0])

    sns.distplot(fitted_data, hist=False, kde=True,
                 kde_kws={'shade': True, 'linewidth': 2},
                 label="Normal", color="green", ax=ax[1])

    plt.legend(loc="best")
    fig.set_figheight(5)
    fig.set_figwidth(10)

    print(f"Lambda value used for Transformation: {fitted_lambda}")


def get_scaler(scaling_option="standard"):
    # TODO finetune this as a hyperparam
    if scaling_option == "standard":
        return StandardScaler()
    elif scaling_option == "minmax":
        return MinMaxScaler()
    else:
        raise Exception("Wrong scaling option given!")


def load_dim_reducer(dim_red_option, n_components):
    if dim_red_option == "PCA":
        dim_reducer = PCA(n_components=n_components)
    elif dim_red_option == "TSNE":
        dim_reducer = TSNE(n_components=n_components)
    elif dim_red_option == "FA":
        dim_reducer = FactorAnalysis(n_components=n_components)
    elif dim_red_option == "SVD":
        dim_reducer = TruncatedSVD(n_components=n_components)
    elif dim_red_option == "autoencoder":
        dim_reducer = load_from_chekpoint(n_components=n_components)
    else:
        raise Exception("wrong dimensionality reduction option given!")
    return dim_reducer


def get_class_weight(labels):
    class_weight_dict = dict()
    for label in labels:
        if label not in class_weight_dict.keys():
            class_weight_dict[label] = 1
        else:
            class_weight_dict[label] += 1
    num_labels = len(labels)
    for (key, value) in class_weight_dict.items():
        class_weight_dict[key] = num_labels / class_weight_dict[key]
    return class_weight_dict
