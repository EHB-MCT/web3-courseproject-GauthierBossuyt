import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.construct import rand
import yaml
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

playlist_tracks = pd.read_pickle("music/data/playlist_tracks.pkl")
recommendation_tracks = pd.read_pickle("music/data/recommendation_tracks.pkl")

playlist_tracks = playlist_tracks.drop_duplicates(subset = 'id', keep = 'first').reset_index()
playlist_tracks.to_excel('playlist_tracks.xlsx')
recommendation_tracks = recommendation_tracks.drop_duplicates(subset = 'id', keep = 'first').reset_index()
recommendation_tracks = recommendation_tracks[~recommendation_tracks['id'].isin(playlist_tracks['id'].tolist())]

with open('music/data/playlists.yml', 'r') as stream:
    playlists = yaml.safe_load(stream)

playlist_tracks['ratings'] = playlist_tracks['playlist_id'].apply(lambda x: 1 if x in playlists.values() else 0)
x = playlist_tracks[['popularity', 'explicit', 'duration_ms', 'danceability', 'energy',
                        'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
                        'liveness', 'valence', 'tempo', 'time_signature', 'genres']]

y = playlist_tracks['ratings']

x = x.dropna()
recommendation_tracks = recommendation_tracks.dropna()

x = x.drop('genres', 1).join(x['genres'].str.join('|').str.get_dummies())
x_recommend = recommendation_tracks.copy()
x_recommend = x_recommend.drop('genres', 1).join(x_recommend['genres'].str.join('|').str.get_dummies())

x = x[x.columns.intersection(x_recommend.columns)]
x_recommend = x_recommend[x_recommend.columns.intersection(x.columns)]
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size = 0.2, random_state = 42)



rf = RandomForestClassifier(n_estimators= 1000, random_state= 42)
rfecv = RFECV(estimator=rf, step=1, n_jobs=-1, cv=StratifiedKFold(2), verbose = 1, scoring='roc_auf')