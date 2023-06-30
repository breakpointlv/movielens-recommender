import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from fastai.callback.tracker import EarlyStoppingCallback
from fastai.collab import CollabDataLoaders, collab_learner, TabularCollab
from fastai.data.block import TransformBlock
from fastai.data.transforms import RandomSplitter
from fastai.learner import load_learner
from fastai.tabular.core import Categorify
from fastcore.basics import range_of
import numpy as np
from torch import tensor

df_ratings = pd.read_csv('data/ratings.csv')
df_movies = pd.read_csv('data/movies.csv')
df_tags = pd.read_csv('data/tags.csv')

df_ratings = df_ratings.drop(columns=['timestamp'])
df_tags = df_tags.drop(columns=['timestamp'])

df_ratings = df_ratings.merge(df_movies, on='movieId')

n_factors = 50
bs = 2048
epochs = 5
lr = 0.005
wd = 0.1

dls = CollabDataLoaders.from_df(
    df_ratings,
    user_name='userId',
    item_name='movieId',
    rating_name='rating',
    bs=bs,
    seed=42
)

learn = collab_learner(
    dls,
    n_factors=n_factors,
    y_range=(0, 5.5),
    use_nn=False
)

print('Starting training...')
learn.fit_one_cycle(
    epochs,
    lr,
    wd=wd,
    cbs=EarlyStoppingCallback(patience=2)
)

learn.export('models/collab_learner.pkl')
print('Model saved!')