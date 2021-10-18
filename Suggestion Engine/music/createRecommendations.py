import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import yaml

from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, normalize

import spotipy
from spotipy.oauth2 import SpotifyOAuth

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
#Error np.matrix is deprecated --> np.asarray

playlist = 'EDM'


playlist_tracks_df = pd.read_pickle('music/data/playlist_tracks.pkl')
playlist_tracks_df['popularity'] = playlist_tracks_df['popularity'] / 100 # Normalized to  0 - 1
playlist_tracks_df.head()

with open('music/data/playlists.yml', 'r') as stream:
    playlist_ids = yaml.safe_load(stream)


def get_interacted_tracks(tracks, playlist_id, drop_duplicates=True):
        interacted_track_ids = set(tracks[tracks['playlist_id'] == playlist_id]['id'])
        tracks_interacted = tracks[tracks['id'].isin(interacted_track_ids)]
        tracks_not_interacted = tracks[~tracks['id'].isin(interacted_track_ids)]

        if drop_duplicates is True:
            tracks_interacted = tracks_interacted.drop_duplicates(subset='id', keep="first").reset_index()
            tracks_not_interacted = tracks_not_interacted.drop_duplicates(subset='id', keep="first").reset_index()
        
        return tracks_interacted, tracks_not_interacted


class ModelEvaluator:

    def __init__(self, tracks):
        self.tracks = tracks
    
    
    def evaluate_model_for_playlist(self, model, playlist_id, n=100, seed=42):
        tracks_interacted, tracks_not_interacted = get_interacted_tracks(self.tracks, playlist_id)
        train, test = train_test_split(tracks_interacted, test_size=0.2, random_state=seed)
        ranked_recommendations_df = model.recommend_tracks(playlist_id)

        hits_at_5_count, hits_at_10_count = 0,0
        for index, row in test.iterrows(): #iterates over a dataframe's rows and columns
            non_interacted_sample = tracks_not_interacted.sample(n, random_state=seed) #sample returns a list of random items of given length 
            evaluation_ids = [row['id']] + non_interacted_sample['id'].tolist()
            evaluation_recommendations_df = ranked_recommendations_df[ranked_recommendations_df['id'].isin(evaluation_ids)]
            hits_at_5_count += 1 if row['id'] in evaluation_recommendations_df['id'][:5].tolist() else 0
            hits_at_10_count += 1 if row['id'] in evaluation_recommendations_df['id'][:10].tolist() else 0
        
        playlist_metrics = {'n':n,
                            'evaluation_count': len(test),
                            'hits@5': hits_at_5_count,
                            'hits@10': hits_at_10_count,
                            'recall@5': hits_at_5_count / len(test),
                            'recall@10': hits_at_10_count / len(test)
        }

        return playlist_metrics

    def evaluate_model(self, model, n=100, seed=42):
        playlists = []
        for playlist_id in self.tracks['playlist_id'].unique():
            playlist_metrics = self.evaluate_model_for_playlist(model, playlist_id, n=n, seed=seed)
            playlist_metrics['playlist_id'] = playlist_id
            playlists.append(playlist_metrics)
        
        detailed_playlists_metrics = pd.DataFrame(playlists).sort_values('evaluation_count', ascending=False)

        global_recall_at_5 = detailed_playlists_metrics['hits@5'].sum() / detailed_playlists_metrics['evaluation_count'].sum()
        global_recall_at_10 = detailed_playlists_metrics['hits@10'].sum() / detailed_playlists_metrics['evaluation_count'].sum()

        global_metrics = {'model_name': model.model_name,
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10
                        }
        return global_metrics, detailed_playlists_metrics


class PopularityRecommender:
    
    def __init__(self, tracks):
        self.tracks = tracks
        self.model_name = 'Popularity Recommender'
    
    def recommend_tracks(self, playlist_id, ignore_ids = []):
        recommendations_df = self.tracks[~self.tracks['id'].isin(ignore_ids)] \
                                .drop_duplicates(subset='id', keep='first').reset_index() \
                                .sort_values('popularity', ascending=False)
        
        return recommendations_df


class ContentRecommender:
    def __init__(self,tracks, ngram_range=(1,2), min_df = 0.003, max_df = 0.5, max_features = 5000):
        self.tracks = tracks
        self.matrix = 0
        self.feature_names = 0
        self.ngram_range = ngram_range
        self.min = min_df
        self.max = max_df
        self.max_features = max_features
        self.model_name = 'Content-based Recommender'

    def setMatrix(self):
        self.tracks['genres_str'] = self.tracks['genres'].apply(lambda x: ' '.join(x))
        
        vectorizer = TfidfVectorizer(analyzer = 'word', ngram_range = self.ngram_range, min_df = self.min, max_df = self.max, max_features = self.max_features, stop_words=stopwords.words('english'))

        matrix = vectorizer.fit_transform(self.tracks['name'] + ' ' +
                                            self.tracks['artist_name'] + ' ' +
                                            self.tracks['album_name'] + ' ' +
                                            self.tracks['playlist_name'] + ' ' +
                                            self.tracks['genres_str']
                                           )
        
        feature_names = vectorizer.get_feature_names_out()

        self.matrix = matrix
        self.feature_names = feature_names 

    def build_playlists_profiles(self):
        playlist_profiles = {}
        for playlist_id in self.tracks['playlist_id'].unique():
            
            interacted_tracks, non_interacted_tracks = get_interacted_tracks(self.tracks, playlist_id, drop_duplicates=False)
            playlist_profiles[playlist_id] = self.build_playlists_profile(playlist_id, interacted_tracks.set_index('playlist_id'))

        return playlist_profiles
    
    def build_playlists_profile(self, playlist_id, interactions_indexed_df):
        interaction_tracks_df = interactions_indexed_df.loc[playlist_id]
        playlist_track_profiles = self.get_track_profiles(interaction_tracks_df['id'])
    
        playlist_track_profiles_array = np.sum(playlist_track_profiles, axis=0)
        playlist_track_profiles_norm = normalize(playlist_track_profiles_array)
        return playlist_track_profiles_norm
    
    def get_track_profiles(self, track_ids):
        track_profile_list = [self.get_track_profile(x) for x in track_ids]
        track_profiles = vstack(track_profile_list)
        return track_profiles
        
    def get_track_profile(self, track_id):
        idx = self.tracks['id'].tolist().index(track_id)
        track_profile = self.matrix[idx: idx + 1]
        return track_profile

    def get_similar_tracks(self, playlist_id):
        playlist_profile = self.build_playlists_profiles()
        cosine_similarities =  cosine_similarity(playlist_profile[playlist_id], self.matrix)
        similar_indices = cosine_similarities.argsort().flatten()
        similar_tracks = sorted([(self.tracks['id'].tolist()[i], cosine_similarities[0,i]) for i in similar_indices], key = lambda x: -x[1])
        return similar_tracks

    def recommend_tracks(self, playlist_id, ignore_ids = []):
        self.setMatrix()
        similar_tracks = self.get_similar_tracks(playlist_id)
        similar_tracks_non_interacted = list(filter(lambda x: x[0] not in ignore_ids, similar_tracks))
        recommendations_df = pd.DataFrame(similar_tracks_non_interacted, columns=['id', 'recStrength']) \
                                        .drop_duplicates(subset='id', keep='first').reset_index() \
                                        .sort_values('recStrength', ascending=False)
        recommendations_df_info = pd.merge(recommendations_df, self.tracks.drop_duplicates(subset='id', keep='first'), how='left', on='id')

        return recommendations_df_info


class CollaborativeRecommender:

    def __init__(self, tracks):
        self.tracks = tracks
        self.matrix = 0
        self.model_name = 'Collaborative Recommender'
    
    def createMatrix(self):
        self.tracks['event_strength'] = 1
        tracks_matrix_df = self.tracks.pivot_table(index='playlist_id', columns='id', values='event_strength', aggfunc='sum').fillna(0)
        tracks_matrix = tracks_matrix_df.values
        tracks_sparse = csr_matrix(tracks_matrix)
        u, s, vt = svds(tracks_sparse, k=15)
        s = np.diag(s)
        tracks_predict_ratings = np.dot(np.dot(u,s), vt)
        tracks_predict_ratings_norm = (tracks_predict_ratings - tracks_predict_ratings.min())/ (tracks_predict_ratings.max() - tracks_predict_ratings.min())
        matrix_preds_df = pd.DataFrame(tracks_predict_ratings_norm, columns= tracks_matrix_df.columns, index=self.tracks['playlist_id'].unique()).transpose()
        self.matrix = matrix_preds_df

    def recommend_tracks(self, playlist_id, ignore_ids = []):
        self.createMatrix()
        sorted_playlist_predictions = self.matrix[playlist_id].sort_values(ascending=False)\
                                                    .reset_index().rename(columns={playlist_id: 'recStrength'})

        recommendations_df = sorted_playlist_predictions[~sorted_playlist_predictions['id'].isin(ignore_ids)]\
                                                        .drop_duplicates(subset='id', keep='first').reset_index()\
                                                        .sort_values('recStrength', ascending = False)

        recommendations_df_info = pd.merge(recommendations_df, self.tracks.drop_duplicates(subset='id', keep='first'), how='left', on='id')
        return recommendations_df_info     
    

class HybridRecommender:
    def __init__(self, tracks, content_model, collaborative_model, content_weight = 1, collaborative_weight = 2, popularity_weight = 1):
        self.tracks = tracks
        self.model_name = 'Hybrid Recommender'
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.popularity_model = popularity_model

        #Relative weights
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.popularity_weight = popularity_weight


    def recommend_tracks(self, playlist_id, ignore_ids = []):

        content_recs_df = self.content_model.recommend_tracks(playlist_id, ignore_ids)\
                                    .rename(columns = {'recStrength' : 'recStrengthContent'})
        
        collaborative_recs_df = self.collaborative_model.recommend_tracks(playlist_id, ignore_ids)\
                                    .rename(columns = {'recStrength' : 'recStrengthCollaborative'})
        
        combined_recs_df = content_recs_df.merge(collaborative_recs_df, how = 'outer', on = 'id').fillna(0)
        
        combined_recs_df['recStrengthHybrid'] = (combined_recs_df['recStrengthContent'] * self.content_weight) \
                                                + (combined_recs_df['recStrengthCollaborative'] * self.collaborative_weight)

        recommendations_df = combined_recs_df \
                                .drop_duplicates(subset = 'id', keep = 'first').reset_index() \
                                .sort_values('recStrengthHybrid', ascending = False)
        
        recommendations_df = pd.merge(recommendations_df, self.tracks.drop_duplicates(subset = 'id', keep = 'first'), how = 'left', on = 'id')
        
        return recommendations_df


class HybridPopularityRecommender:
    def __init__(self, tracks, content_model, collaborative_model, popularity_model, content_weight = 1, collaborative_weight = 2, popularity_weight = 1):
        self.tracks = tracks
        self.model_name = 'HybridPopularity Recommender'
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.popularity_model = popularity_model

        #Relative weights
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.popularity_weight = popularity_weight


    def recommend_tracks(self, playlist_id, ignore_ids = []):

        content_recs_df = self.content_model.recommend_tracks(playlist_id, ignore_ids)\
                                    .rename(columns = {'recStrength' : 'recStrengthContent'})
        
        collaborative_recs_df = self.collaborative_model.recommend_tracks(playlist_id, ignore_ids)\
                                    .rename(columns = {'recStrength' : 'recStrengthCollaborative'})

        popularity_recs_df = self.popularity_model.recommend_tracks(playlist_id, ignore_ids)\
                                    .rename(columns = {'popularity' : 'recStrengthPopularity'})
        
        combined_recs_df = content_recs_df.merge(collaborative_recs_df, how = 'outer', on = 'id')\
                                          .merge(popularity_recs_df, how = 'outer', on = 'id').fillna(0)
        
        combined_recs_df['recStrengthHybridPopularity'] = (combined_recs_df['recStrengthContent'] * self.content_weight) \
                                                        + (combined_recs_df['recStrengthCollaborative'] * self.collaborative_weight)\
                                                        + (combined_recs_df['recStrengthPopularity'] * self.popularity_weight)

        recommendations_df = combined_recs_df \
                                .drop_duplicates(subset = 'id', keep = 'first').reset_index() \
                                .sort_values('recStrengthHybridPopularity', ascending = False)
        
        recommendations_df = pd.merge(recommendations_df, self.tracks.drop_duplicates(subset = 'id', keep = 'first'), how = 'left', on = 'id')
        
        return recommendations_df


interacted_tracks, non_interacted_tracks = get_interacted_tracks(playlist_tracks_df, playlist_ids)

#MODELEVALUATOR
model_evaluator = ModelEvaluator(playlist_tracks_df)


#POPULARITY RECOMMENDING SYSTEM
popularity_model = PopularityRecommender(playlist_tracks_df)
# popularity_recommendations = popularity_model.recommend_tracks(playlist_ids[playlist], interacted_tracks['id'].tolist())

# popularity_model_metrics, popularity_model_details = model_evaluator.evaluate_model(popularity_model)
# print(popularity_model_metrics)

#CONTENT-BASED RECOMMENDING SYSTEM
content_model = ContentRecommender(playlist_tracks_df)
content_model_recommendations = content_model.recommend_tracks(playlist_ids[playlist], interacted_tracks['id'].tolist())

# content_model_metrics, content_model_details = model_evaluator.evaluate_model(content_model)
# print(content_model_metrics)


#COLLABORATIVE RECOMMENDER
collaborative_model = CollaborativeRecommender(playlist_tracks_df)
# collaborative_model_recommendations = collaborative_model.recommend_tracks(playlist_ids[playlist], interacted_tracks['id'].tolist())

# collaborative_model_metrics, collaborative_model_details = model_evaluator.evaluate_model(collaborative_model)
# print(collaborative_model_metrics)


#HYBRID RECOMMENDER
hybrid_model = HybridRecommender(playlist_tracks_df, content_model, collaborative_model)
# hybrid_model_recommendations = hybrid_model.recommend_tracks(playlist_ids[playlist], interacted_tracks['id'].tolist())

# hybrid_model_metrics, hybrid_model_details = model_evaluator.evaluate_model(hybrid_model)
# print(hybrid_model_metrics)


#HYBRID POPULARITY RECOMMENDER
hybridpopularity_model = HybridPopularityRecommender(playlist_tracks_df, content_model, collaborative_model, popularity_model)
# hybridpopularity_model_recommendations = hybridpopularity_model.recommend_tracks(playlist_ids[playlist], interacted_tracks['id'].tolist())

# hybridpopularity_model_metrics, hybridpopularity_model_details = model_evaluator.evaluate_model(hybridpopularity_model)
# print(hybridpopularity_model_metrics)



# with pd.ExcelWriter('music/recommendations.xlsx') as writer:
#     popularity_recommendations.to_excel(writer, sheet_name = 'Popularity Recommender')
#     content_model_recommendations.to_excel(writer, sheet_name = 'Content Recommender')
#     collaborative_model_recommendations.to_excel(writer, sheet_name = 'Collaboratve Recommender')
#     hybrid_model_recommendations.to_excel(writer, sheet_name = 'Hybrid Recommender')
#     hybridpopularity_model_recommendations.to_excel(writer, sheet_name = 'Hybrid Popularity Recommender')

# recommended_tracks = hybridpopularity_model_recommendations[hybridpopularity_model_recommendations['recStrengthHybridPopularity'] >= 1.15]['id']
recommended_tracks = content_model_recommendations[content_model_recommendations['recStrength'] >= 0.5]['id']
