from sklearn.metrics import mean_squared_error
from statistics import mean

def get_contents_based_similarity(datapoint_1, datapoint_2, distance="mse"):
    return mean_squared_error(datapoint_1, datapoint_2)

def get_users_based_similarity(datapoints_1, datapoints_2):
    sims = []
    for i in range(len(datapoints_1)):
        for j in range(len(datapoints_2)):
            sims.append(datapoints_1[i], datapoints_2[j])
    return mean(sims)

def get_top_n_closes(datapoint, candidates, n=5):
    similarities = [get_contents_based_similarity(datapoint, candidate) for candidate in candidates]
    candidates = list(zip(candidates, similarities))
    candidates = sorted(candidates, key=lambda x: (-1) * x[1])
    # TODO idea: deployment with flask/streamlit/tornado app
    # could be inspired from here
    # https://share.streamlit.io/ruthgn/beer-recommender/main/beer-recommender-app.py
    return candidates[:n]
