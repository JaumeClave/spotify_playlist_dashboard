import streamlit as st
import pandas as pd
import numpy as np
import spotipy
import spotipy.util as util
from dateutil.parser import parse as parse_date
import plotly.graph_objects as go
from pylab import *
import calplot

################################################## VARIABLES ###########################################################

FILE_PATH_API_CREDENTIAL = r"C:\Users\Jaume\Documents\Python Projects\spotify_compare\data\external\api_keys.csv"
SPOTIPY_SCOPE = "ugc-image-upload user-read-playback-state streaming user-read-email playlist-read-collaborative \
    user-modify-playback-state user-read-private playlist-modify-public user-library-modify user-top-read \
    user-read-playback-position user-read-currently-playing playlist-read-private user-follow-read \
    app-remote-control user-read-recently-played playlist-modify-private user-follow-modify user-library-read"
ARTIST = "artist"
ID = "id"
SONGS = "songs"
ALBUM = "album"
COUNT = "count"
ID = "id"
ADDED_AT_COLUMN = "added_at"
DAY_FREQUENCY = "D"
START_OF_YEAR = "-1-1"
END_OF_YEAR = "-12-31"
COLORWAY_CALENDAR = 'PRGn'
GLOBAL_TOP50_PLAYLIST_LINK = "https://open.spotify.com/playlist/37i9dQZEVXbMDoHDwVN2tF?si=e7a63efae3db4169"

################################################## FUNCTIONS ###########################################################

def make_api_cred_dataframe(file_path):
    """
    Function loads the csv file containing Spotify Developer client id/secret
    :param file_path:
    :return:
    """
    api_creds = pd.read_csv(file_path, header=None)
    return api_creds

def make_tokens_from_api_creds(api_creds_df):
    """
    Function gets user_id, client_id, client_secret
    :param api_creds_df:
    :return:
    """
    user_id = api_creds_df[1].iloc[0]
    client_id = api_creds_df[1].iloc[1]
    client_secret = api_creds_df[1].iloc[2]
    return user_id, client_id, client_secret

def get_token_authentication(user_id, client_id, client_secret, scope):
    """
    Function gets token user to authenticate
    :param user_id:
    :param client_id:
    :param client_secret:
    :param scope:
    :return:
    """
    token = util.prompt_for_user_token(
        user_id, scope, client_id = client_id, client_secret = client_secret, redirect_uri='http://localhost/')
    sp = spotipy.Spotify(auth = token)
    return sp

def pipeline_spotipy_authentication():
    """
    Function pipelines the workflow needed to authenticate on Spotify
    :return:
    """
    # api_creds = make_api_cred_dataframe(FILE_PATH_API_CREDENTIAL) # Using Streamlit Secrets Management
    # user_id, client_id, client_secret = make_tokens_from_api_creds(api_creds)
    user_id, client_id, client_secret = st.secrets["user_id"], st.secrets["client_id"], st.secrets["client_secret"]
    sp = get_token_authentication(user_id, client_id, client_secret, SPOTIPY_SCOPE)
    return sp, user_id

def get_playist_id_from_link(playlist_link):
    """
    Function returns the Spotify playlist id from the Spotify generated link
    :param playlist_link:
    :return:
    """
    id = playlist_link.split("/")[-1].split("?")[0]
    return id

def get_playlist_name_description_image_url(playlist_id):
    """
    Function returns the associated playlist cover art
    :param playlist_id:
    :return:
    """
    playlist = sp.user_playlist(user_id, playlist_id)
    name = playlist["name"]
    description = playlist['description']
    image_url = playlist["images"][0]["url"]
    collaborative = playlist['collaborative']
    followers = playlist["followers"]["total"]
    owner = playlist["owner"]["display_name"]
    public = playlist["public"]
    return name, description, image_url, collaborative, followers, owner, public

def get_playlist_tracks(username, playlist_id):
    """
    Function returns all songs (JSON) for a playlist. Spotipy returns paginated results of 100 songs and this function filters through all pages
    :param username:
    :param playlist_id:
    :return:
    """
    results = sp.user_playlist_tracks(username,playlist_id)
    tracks = results['items']
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    return tracks

def get_playlist_df(playlist_id):
    """
    Function creates a dataframe containing songs and relevant song details from a playlist
    :param playlist_id:
    :return:
    """
    playlist = sp.user_playlist(user_id, playlist_id)
    tracks = get_playlist_tracks(user_id, playlist_id)
    next_uri = playlist['tracks']['next']
    name = playlist['name']
    tracks_df = pd.DataFrame([(track['track']['id'], track['track']['name'],
                               track["track"]["artists"][0]["id"], track['track']['artists'][0]['name'],
                               track["track"]["album"]["id"], track['track']['album']['name'],
                               track['track']['popularity'],
                               parse_date(track['track']['album']['release_date'])
                               if track['track']['album']['release_date'] else None,
                               parse_date(track['added_at']))
                              for track in tracks],
                             columns=['id', 'name', "artist_id", 'artist', "album_id", 'album', 'track popularity', 'release_date', 'added_at'])
    tracks_df['playlist_name'] = name
    tracks_df["release_year"] = tracks_df["release_date"].dt.year
    return tracks_df

def get_playlist_count_songs_artists(playlist_df):
    """
    Function gets the amount of songs and count of artists in a playlist dataframe
    :param playlist_df:
    :return:
    """
    count_songs = len(playlist_df["name"])
    count_artists = len(playlist_df["artist"].unique())
    return count_songs, count_artists

def make_artist_in_playlist_count_df(playlist_df):
    """
    Function makes a dataframe showing the amount of songs each artist in the playlist has in the playlist
    :param playlist_df: dataframe of the playlist
    :return: artist count dataframe
    """
    artist_count_df = playlist_df.groupby(ARTIST).count()[ID].reset_index().sort_values(ID, ascending=False).rename(columns={ID: COUNT})
    artist_count_df.set_index(ARTIST, inplace=True)
    artist_count_df.index.name = ARTIST
    return artist_count_df

def make_album_in_playlist_count_df(playlist_df):
    """
    Function makes a dataframe showing the amount of songs each album has in the playlist has in the playlist
    :param playlist_df: dataframe of the playlist
    :return: artist count dataframe
    """
    album_count_df = playlist_df.groupby(ALBUM).count()[ID].reset_index().sort_values(ID, ascending=False).rename(columns={ID: COUNT})
    album_count_df.set_index(ALBUM, inplace=True)
    album_count_df.index.name = ALBUM
    return album_count_df

def pipeline_make_count_artist_album_df(playlist_df):
    """

    :param playlist_df:
    :return:
    """
    artist_count_df = make_artist_in_playlist_count_df(playlist_df)
    album_count_df = make_album_in_playlist_count_df(playlist_df)
    return artist_count_df, album_count_df

def make_playlist_track_with_featured_df(playlist_df):
    """

    :return:
    """
    features = list()
    for n, chunk_series in playlist_df.groupby(np.arange(len(playlist_df)) // 100)['id']:
        features += sp.audio_features([*map(str, chunk_series)])
    features_df = pd.DataFrame.from_dict(filter(None, features))
    tracks_with_features_df = playlist_df.merge(features_df, on = ['id'], how = 'inner')
    return tracks_with_features_df

def convertMillis(millis):
    """
    Function returns seconds, minutes and hours with millisecond input
    :param millis:
    :return:
    """
    seconds=(millis/1000)%60
    minutes=(millis/(1000*60))%60
    hours=(millis/(1000*60*60))%24
    return seconds, minutes, hours

def get_hour_minutes(minutes, hours):
    """

    :param minutes:
    :param hours:
    :return:
    """
    str_minutes = str(minutes)[0]
    str_hours = str(hours)[0]
    return str_minutes, str_hours

def pipeline_hour_min_length_of_playlist(tracks_with_features_df):
    """

    :return:
    """
    millis = tracks_with_features_df["duration_ms"].sum()
    seconds, minutes, hours = convertMillis(millis)
    str_minutes, str_hours = get_hour_minutes(minutes, hours)
    return str_minutes, str_hours


def make_global_top_50_mean_feature_value():
    """
    Function calls the Global Top 50 playlist and returns mean values of key features
    :return:
    """
    playlist_df = get_playlist_df(GLOBAL_TOP50_PLAYLIST_LINK)
    top_50_feature_df = make_playlist_track_with_featured_df(playlist_df)
    t50_energy, t50_loudness, t50_tempo, t50_speechiness, t50_valence, t50_danceability = \
        top_50_feature_df["energy"].mean(), top_50_feature_df["loudness"].mean(), top_50_feature_df["tempo"].mean(), \
        top_50_feature_df["speechiness"].mean(), top_50_feature_df["valence"].mean(), \
        top_50_feature_df["danceability"].mean()
    return t50_energy, t50_loudness, t50_tempo, t50_speechiness, t50_valence, t50_danceability


def make_playlist_feature_means(playlist_df):
    """
    Function returns the mean values for key playlist features (energy, loudness, etc...)
    :param playlist_df:
    :return:
    """
    playlist_energy, playlist_loudness, playlist_tempo, playlist_speechiness, playlist_valence, playlist_danceability = \
        playlist_df["energy"].mean(), playlist_df["loudness"].mean(), playlist_df["tempo"].mean(), \
        playlist_df["speechiness"].mean(), playlist_df["valence"].mean(), \
        playlist_df["danceability"].mean()
    return playlist_energy, playlist_loudness, playlist_tempo, playlist_speechiness, playlist_valence,\
           playlist_danceability

def get_xy_axis_and_data(tracks_with_features_df, yaxis, customdata, xaxis=None):
    """

    :param tracks_with_features_df:
    :param axis:
    :param yaxis:
    :param customdata:
    :return:
    """
    if xaxis is None:
        x_axis = list(tracks_with_features_df.index)
    else:
        x_axis = list(tracks_with_features_df[xaxis])
    y_axis = list(tracks_with_features_df[yaxis])
    customdata_list = list(tracks_with_features_df[customdata])
    return x_axis, y_axis, customdata_list

def plot_scatter(x_axis, y_axis, customdata_list, playlist_mean_line, mean_top50_line):
    """

    :param x_axis:
    :param y_axis:
    :param customdata_list:
    :return:
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode='markers', name='markers',
                             hovertext=customdata_list, hoverlabel=dict(namelength=0), hovertemplate='%{hovertext}<br>Energy: %{y}<br>',
                             marker = dict(size = 8, color = y_axis, colorscale = 'algae', opacity=0.8)))
    fig.update_layout(width = 800, height = 400, margin = dict(l = 0, r = 00, b = 0, t = 0, pad = 2), template = "plotly_dark")
    fig.add_hline(y=playlist_mean_line, line_dash="dash", line_color="green")
    fig.add_hline(y=mean_top50_line, line_dash="dash", line_color="white")
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    st.plotly_chart(fig, use_container_width=True)
    return None

def pipeline_plot_feature(tracks_with_features_df, yaxis, customdata, playlist_mean_line, mean_top50_line, xaxis=None):
    """

    :param playlist_df:
    :param yaxis:
    :param customdata:
    :param xaxis:
    :return:
    """
    x_axis, y_axis, customdata_list = get_xy_axis_and_data(tracks_with_features_df, yaxis, customdata, xaxis)
    plot_scatter(x_axis, y_axis, customdata_list, playlist_mean_line, mean_top50_line)
    return None

def make_daily_add_series(playlist_df):
    """

    :param playlist_df:
    :return:
    """
    dms = playlist_df.groupby(playlist_df[ADDED_AT_COLUMN].dt.to_period(DAY_FREQUENCY)).count()[ID].to_timestamp()
    max_year = playlist_df[ADDED_AT_COLUMN].dt.to_period(DAY_FREQUENCY).max().year
    min_year = playlist_df[ADDED_AT_COLUMN].dt.to_period(DAY_FREQUENCY).min().year
    idx = pd.date_range(str(min_year) + START_OF_YEAR, str(max_year) + END_OF_YEAR)
    dms.index = pd.DatetimeIndex(dms.index)
    daily_adds = dms.reindex(idx, fill_value=0)
    return daily_adds

def plot_date_added_calendar(daily_adds):
    """

    :param daily_adds:
    :return:
    """
    cmap = cm.get_cmap(COLORWAY_CALENDAR, 10)
    fig, ax = calplot.calplot(daily_adds, cmap = cmap)
    st.pyplot(fig)
    return None

def pipeline_date_added_calendar(playlist_df):
    """

    :param playlist_df:
    :return:
    """
    daily_adds = make_daily_add_series(playlist_df)
    plot_date_added_calendar(daily_adds)
    return None


def make_artist_count_df(playlist_df):
    """
    Function returns a dataframe with the artist name and count of times in playlist
    :return:
    """
    artist_count = playlist_df.groupby("artist").count()["name"].reset_index().sort_values("name", ascending=True)
    return artist_count

def make_artist_count_plot(artist_count):
    """
    Function plots the times an artist occurs in a playlist
    :param artist_count:
    :return:
    """
    fig = go.Figure(go.Bar(
        x=artist_count["name"],
        y=artist_count["artist"],
        orientation='h', marker_color="#1DB954"))
    fig.update_layout(width=800, height=600, margin=dict(l=50, r=50, b=50, t=50, pad=4), template="plotly_dark")
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    st.plotly_chart(fig, use_container_width=True)
    return None

def pipeline_artist_count_plot(playlist_df):
    """
    Function pipelines the workflow required to plot artist count
    :param playlist_df:
    :return:
    """
    artist_count = make_artist_count_df(playlist_df)
    make_artist_count_plot(artist_count)
    return None

################################################## STREAMLIT ###########################################################

# st.header("Spotify Compare")
st.set_page_config(page_title='Spotify: Playlist Dashboard',
                   # page_icon='https://pbs.twimg.com/profile_images/' \
                   #           '1265092923588259841/LdwH0Ex1_400x400.jpg',
                   layout="wide")
# Authenticate Spotify

from spotipy_client import *

client_id = 'f7eedef5bb4b4fe4ad5f7b276f8db10c'
client_secret = '14505366a61b431994f7afe58ecdc550'
user_id = "1113039340"

spotify = SpotifyAPI(client_id, client_secret)
access_token = spotify.access_token
sp = spotipy.Spotify(auth = access_token)

############ EHA

Types_of_Features = ("acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo", "valence")

st.title("Spotify Features App")
Name_of_Artist = st.text_input("Artist Name")
Name_of_Feat = st.selectbox("Feature", Types_of_Features)
button_clicked = st.button("OK")


Data = spotify.search({"artist": f"{Name_of_Artist}"}, search_type="track")

need = []
for i, item in enumerate(Data['tracks']['items']):
    track = item['album']
    track_id = item['id']
    song_name = item['name']
    popularity = item['popularity']
    need.append((i, track['artists'][0]['name'], track['name'], track_id, song_name, track['release_date'], popularity))

Track_df = pd.DataFrame(need, index=None, columns=('Item', 'Artist', 'Album Name', 'Id', 'Song Name', 'Release Date', 'Popularity'))

access_token = spotify.access_token

headers = {
    "Authorization": f"Bearer {access_token}"
}
endpoint = "https://api.spotify.com/v1/audio-features/"

Feat_df = pd.DataFrame()
for id in Track_df['Id'].iteritems():
    track_id = id[1]
    lookup_url = f"{endpoint}{track_id}"
    #print(lookup_url)
    ra = requests.get(lookup_url, headers=headers)
    audio_feat = ra.json()
    #print(audio_feat)
    Features_df = pd.DataFrame(audio_feat, index=[0])
    Feat_df = Feat_df.append(Features_df)
    #print(Feat_df)

st.table(Feat_df)

# ROW 1 ------------------------------------------------------------------------

row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.columns(
    (.1, 2, 1.5, 1, .1)
)

row1_1.title('Spotify - Playlist Dashboard')

with row1_2:
    st.write('#')
    row1_2.write(
        'A Web App by [Jaume Clave](https://github.com/JaumeClave)')

# ROW 2 ------------------------------------------------------------------------

row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3 = st.columns(
    (.1, 1.6, .1, 1.6, .1)
)

with row2_1:
    # Playlist link/id to name, image and description
    sptfy_playlist_link = st.text_input("Spotify playlist link - Spotify playlist > Share > Copy link to playlist")

# ROW 3 ------------------------------------------------------------------------

if len(sptfy_playlist_link) != 0:
    playlist_id = get_playist_id_from_link(sptfy_playlist_link)
    playlist_df = get_playlist_df(playlist_id)
    tracks_with_features_df = make_playlist_track_with_featured_df(playlist_df)
    str_minutes, str_hours = pipeline_hour_min_length_of_playlist(tracks_with_features_df)
    name, description, image_url, collaborative, followers, owner, public = get_playlist_name_description_image_url(playlist_id)

    st.write('')
    row1_space1, row1_1, row1_space2, row1_2, row1_space3, row1_3, row1_space4 = st.columns(
        (.15, 1, .4, 1.1, .000000001, 3, 0.15))

    with row1_1:

        st.image(image_url, width=300)

    with row1_2:
        st.subheader(' ')
        st.text(' ')
        st.text(
            f"Name: {name}")
        st.text(
            f"Owner: {owner}")
        st.text(
            f"Songs: {len(playlist_df)}")
        st.text(
            f"Length : {str_hours} hrs {str_minutes} mins")
        st.text(
            f"Collaborative: {collaborative}")
        st.text(
            f"Public: {public}")

    with row1_3:
        st.dataframe(playlist_df[["name", "artist", "album"]])

# ROW 4 ------------------------------------------------------------------------
    row9_spacer1, row9_1, row9_spacer2, = st.columns(
        (.1, 2, 0.000001)
    )
    with row9_1:
        st.header("Playlist Features vs. Global Top 50")
        st.write("The dashed white line shows the average feature value for Spotify's Top 50 Global Playlist. "
                 "The dashed green line shows the average feature value for {}. Definitions of features can be found "
                 "in the 'Additional Info' section at the bottom of the page".format(name))


    playlist_energy, playlist_loudness, playlist_tempo, playlist_speechiness, playlist_valence, \
    playlist_danceability = make_playlist_feature_means(tracks_with_features_df)
    t50_energy, t50_loudness, t50_tempo, t50_speechiness, t50_valence, t50_danceability = \
        make_global_top_50_mean_feature_value()
    row2_space1, row2_1, row2_space2, row2_2, row2_space3, row2_3, row2_space4 = st.columns(
        (.15, 1.5, .00000001, 1.5, .00000001, 1.5, 0.15))
    with row2_1:
        st.subheader('Energy')
        pipeline_plot_feature(tracks_with_features_df, "energy", "name", playlist_energy, t50_energy, xaxis=None)

    with row2_2:
        st.subheader('Loudness')
        pipeline_plot_feature(tracks_with_features_df, "loudness", "name", playlist_loudness, t50_loudness, xaxis=None)

    with row2_3:
        st.subheader('Tempo')
        pipeline_plot_feature(tracks_with_features_df, "tempo", "name", playlist_tempo, t50_tempo, xaxis=None)



# ROW 5 ------------------------------------------------------------------------
    st.write('')
    row2_space1, row2_1, row2_space2, row2_2, row2_space3, row2_3, row2_space4 = st.columns(
        (.15, 1.5, .00000001, 1.5, .00000001, 1.5, 0.15))
    with row2_1:
        st.subheader('Speechiness')
        pipeline_plot_feature(tracks_with_features_df, "speechiness", "name", playlist_speechiness, t50_speechiness, xaxis=None)

    with row2_2:
        st.subheader('Valence')
        pipeline_plot_feature(tracks_with_features_df, "valence", "name", playlist_valence, t50_valence, xaxis=None)

    with row2_3:
        st.subheader('Danceability')
        pipeline_plot_feature(tracks_with_features_df, "danceability", "name", playlist_danceability, t50_danceability, xaxis=None)

# ROW 7 ------------------------------------------------------------------------
    st.write('')
    row2_spacer1, row2_1, row2_spacer2, = st.columns(
        (.1, 2, 0.000001)
    )
    with row2_1:
        st.header("Date Added")
        pipeline_date_added_calendar(playlist_df)

# ROW 8 ------------------------------------------------------------------------
    st.write('')
    row8_spacer1, row8_1, row8_spacer2, = st.columns(
        (.1, 2, 0.000001)
    )
    with row8_1:
        st.header("Artist Count")
        pipeline_artist_count_plot(playlist_df)


    # ROW 6 ------------------------------------------------------------------------
    row6_spacer1, row6_1, row6_spacer2 = st.columns((.1, 3.2, .1))

    with row6_1:
        st.markdown('___')
        about = st.expander('About/Additional Info')
        with about:
            '''
            Thanks for checking out my app! It was built entirely using [Spotify]
            (https://developer.spotify.com/) data. Special thanks to the people behind [Spotipy]
            (https://github.com/plamere/spotipy) and [Streamlit](https://streamlit.io/).
            This is the first time I have ever built something like this, so any comments or feedback is greatly appreciated. I hope you enjoy!
    
            ---
            
            This app is a dashboard that runs an analysis on any desired WR or TE who
            has logged at least 30 total targets in the 2020 season. Player info,
            a game log, and six visualizations of various statistics are displayed
            for the selected player. They are briefly described below:
            
            **Danceability** - Danceability describes how suitable a track is for dancing based on a combination of 
            musical elements including tempo, rhythm stability, beat strength, and overall regularity. 
            A value of 0.0 is least danceable and 1.0 is most danceable.
            
            **Energy** - Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and 
            activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, 
            while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include 
            dynamic range, perceived loudness, timbre, onset rate, and general entropy.
            
            **Loudness** - The overall loudness of a track in decibels (dB). Loudness values are averaged across the 
            entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound 
            that is the primary psychological correlate of physical strength (amplitude). Values typical range between 
            -60 and 0 db.
            
            **Speechiness** - Speechiness detects the presence of spoken words in a track. 
            The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the 
            attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values 
            between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or 
            layered, including such cases as rap music. Values below 0.33 most likely represent music and other 
            non-speech-like tracks.
            
            **Tempo** - The overall estimated tempo of a track in beats per minute (BPM). In musical 
            terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
            
            **Valence** - A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks 
            with high valence sound more positive (e.g. happy, cheerful, euphoric) while tracks with low valence sound 
            more negative (e.g. sad, depressed, angry).
            
            ### Jaume Clave, 2021
            '''

#%%
