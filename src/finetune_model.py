import keras_tuner as kt
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def finetune_model(model, data, targets, finetune_option="sk_grid_search", distributions: dict = dict()):
    # TODO finetune the option from the 3 availables ones
    """
    :param model:
    :param data:
    :param targets:
    :param finetune_option:
    :param distributions: dict where keys must be model params and values must be a list of values from where to draw possible values
    :return: if an sk based model is used a set of params is returned, otherwise, a model is returned if we want to finetune a keras model
    """
    RANDOM_SEED = 13
    if finetune_option == "sk_grid_search":
        finetuner = GridSearchCV(model, distributions, verbose = 1)#, random_state=RANDOM_SEED)
    elif finetune_option == "sk_random_search":
        finetuner = RandomizedSearchCV(model, distributions, verbose = 1)#, random_state=RANDOM_SEED)
    else:
        raise Exception("Wrong finetuning option given!")

    if finetune_option.startswith("sk"):
        finetuner.fit(data, targets)
        return finetuner
    elif finetune_option.startswith("keras"):
        finetuner.search(data, targets, epochs=5)
        return finetuner.get_best_models()[0]
