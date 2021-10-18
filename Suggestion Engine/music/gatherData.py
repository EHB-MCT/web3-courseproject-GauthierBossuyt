import spotipy
import yaml
import json
import data_functions as data
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd

spotify_details = open('music/spotify_details.json', "r")
spotify_details = json.load(spotify_details)
scope = "user-library-read user-follow-read user-top-read playlist-read-private"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=spotify_details['client_id'], 
    client_secret=spotify_details['client_secret'], 
    redirect_uri=spotify_details['redirect_uri'], 
    scope = scope,
))

# print("Collecting top artist data...")
# top_artist = data.offset_api_limit(sp, sp.current_user_top_artists()) 
# top_artist = data.get_artist_create_df(top_artist)
# top_artist.to_pickle("music/data/top_artists.pkl")

# print("Collect following artist data...")
# follow_artists = data.offset_api_limit(sp,sp.current_user_followed_artists())
# follow_artists = data.get_artist_create_df(follow_artists)
# follow_artists.to_pickle("music/data/follow_artists.pkl")

# print("Collect top tracks data...")
# top_tracks = data.offset_api_limit(sp, sp.current_user_top_tracks())
# top_tracks = data.get_tracks_create_df(top_tracks)
# top_tracks = data.get_audio_df(sp, top_tracks)
# top_tracks.to_pickle("music/data/top_tracks.pkl")

# print("Collect saved tracks data...")
# saved_tracks = data.offset_api_limit(sp, sp.current_user_saved_tracks())
# saved_tracks = data.get_tracks_create_df(saved_tracks)
# saved_tracks = data.get_audio_df(sp,saved_tracks)
# saved_tracks.to_pickle("music/data/saved_tracks.pkl")

print("Collect playlist tracks data...")
playlist_tracks = data.get_all_playlist_tracks_df(sp, sp.current_user_playlists())
playlist_tracks = data.get_audio_df(sp, playlist_tracks)
playlist_tracks.to_pickle("music/data/playlist_tracks.pkl")

playlist_tracks = pd.read_pickle('music/data/playlist_tracks.pkl')

playlist_dict = dict(zip(playlist_tracks['playlist_name'], playlist_tracks['playlist_id']))
with open('music/data/playlists.yml', 'w') as outfile:
     yaml.dump(playlist_dict, outfile, default_flow_style=False)


recommendation_tracks = data.get_recommendations(sp, playlist_tracks[playlist_tracks['playlist_name']
    .isin(["EDM", "EDM '20", "EDM '19", "EDM '18", "Your Top Songs 2020", "Your Top Songs 2019", "Your Top Songs 2018"
     ])].drop_duplicates(subset='id', keep='first')['id'].tolist())
print('tracks gathered')
recommendation_tracks = data.get_tracks_create_df(recommendation_tracks)
print('df created of tracks')
recommendation_tracks = data.get_audio_df(sp, recommendation_tracks)
print('audio information gathered')
recommendation_tracks.to_pickle("music/data/recommendation_tracks.pkl")

