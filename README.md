# EDA-BeerProfileRating


Take the public dataset from Kaggle: https://www.kaggle.com/ruthgn/beer-profile-and-ratings-data-set.

Undergo an exploratory data analysis in order to understand better the data and have an overall image of what it could offer.

Create at lest 4 models ( that must be further trained) to predict something considering the given dataset.

Write documentation and explain the whole process. 

# Running code

In order to rune the code from this repo you should firstly create an anaconda environment with a version of Python 3.10:

conda create -n eda_env python=3.10

Then, activate the environment from the terminal via: 

conda activate eda_env

Install the necessary packages from the root of the project with:

pip install -r requirements.xt

Then run:

python src/main.py

# Folder system imports problem

If you will encounter any errors related to importing stuff from other Python files from the "src" folder we recommend the following line of code:

export PYTHONPATH=`pwd`

Should fix any path problems.
