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

print('getting the data...')
df_ratings = pd.read_csv('data/ratings.csv')
df_movies = pd.read_csv('data/movies.csv')

df_ratings = df_ratings.drop(columns=['timestamp'])

# one hot encode genres
genre_dummies = df_movies['genres'].str.get_dummies(sep='|')
# covert all columns to int8
genre_dummies = genre_dummies.astype('int8')
df_movies = pd.concat([df_movies, genre_dummies], axis=1)

# add year from title
df_movies['year'] = df_movies['title'].str.extract(r'\((\d{4})\)')
df_movies['year'] = pd.to_numeric(df_movies['year'], errors='coerce')
df_movies['year'] = df_movies['year'].fillna(0).astype('category')

df_movies.drop(columns=['title', 'genres', '(no genres listed)'], inplace=True)
df_ratings = df_ratings.merge(df_movies, on='movieId')

n_factors = 50
bs = 8192
epochs = 10
lr = 0.005
wd = 0.1

# list column names
print(df_ratings.columns)

splits = RandomSplitter(valid_pct=0.2, seed=42)(range_of(df_ratings))
to = TabularCollab(
    df_ratings,
    [Categorify],
    list(df_ratings.columns),
    user_name='userId',
    y_names=['rating'],
    y_block=TransformBlock(),
    splits=splits
)
print('creating dataloaders...')
dls = to.dataloaders(
    path='.',
    bs=bs,
    num_workers=0,
    pin_memory=True
)
print(dls.one_batch())
print('creating learner...')
learn = collab_learner(
    dls,
    n_factors=n_factors,
    y_range=(0, 5.5),
    use_nn=True,
    layers=[256, 128, 64]
)

print('Starting training...')
learn.fit_one_cycle(
    epochs,
    lr,
    wd=wd,
    cbs=EarlyStoppingCallback(patience=2)
)

model_name = 'models/collab_learner_nn.pkl'
learn.export(model_name)
print('Model saved to', model_name, '!')