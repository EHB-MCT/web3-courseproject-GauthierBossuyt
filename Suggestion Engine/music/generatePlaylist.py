import spotipy
import json
from createRecommendations import recommended_tracks
from spotipy.oauth2 import SpotifyOAuth

spotify_details = open('music/spotify_details.json', "r")
spotify_details = json.load(spotify_details)
scope = "playlist-modify-private"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=spotify_details['client_id'], 
    client_secret=spotify_details['client_secret'], 
    redirect_uri=spotify_details['redirect_uri'], 
    scope = scope,
))

new_playlist = sp.user_playlist_create(user = spotify_details['user'],
                                        name = 'Recommender System Playlist',
                                        public=False,
                                        collaborative=False,
                                        description='Created by following the tutorial created on https://github.com/anthonyli358/spotify-recommender-systems'    
                                    )

for id in recommended_tracks:
    sp.user_playlist_add_tracks(user = spotify_details['user'],
                                playlist_id = new_playlist['id'],
                                tracks=[id]
                                )

