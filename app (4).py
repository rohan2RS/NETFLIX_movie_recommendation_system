import pandas as pd
import numpy as np
import pickle
import streamlit as st

# Load the data
data = pd.read_csv('/content/sample_data/netflix_titles.csv')

# Data preprocessing
data['director'].fillna('Unknown', inplace=True)
data['cast'].fillna('Unknown', inplace=True)
data['country'].fillna('Unknown', inplace=True)
data['date_added'].fillna('Not Available', inplace=True)
data['rating'].fillna('Unknown', inplace=True)

data.loc[(data['type'] == 'Movie') & (data['duration'].isnull()), 'duration'] = 'Unknown Duration'
data.loc[(data['type'] == 'TV Show') & (data['duration'].isnull()), 'duration'] = 'Unknown Seasons'

data['date_added'] = pd.to_datetime(data['date_added'], errors='coerce')
data['year_added'] = data['date_added'].dt.year
data['month_added'] = data['date_added'].dt.month

data['date_added'].fillna(pd.Timestamp('1900-01-01'), inplace=True)
data['year_added'].fillna(1900, inplace=True)
data['month_added'].fillna(1, inplace=True)

data['year_added'] = data['year_added'].astype(int)
data['month_added'] = data['month_added'].astype(int)

movie_duration = data[data['type'] == 'Movie'].copy()
movie_duration['duration'] = movie_duration['duration'].astype(str).str.replace(' min', '', regex=False)
movie_duration['duration'] = pd.to_numeric(movie_duration['duration'], errors='coerce')
movie_duration['duration'].fillna(movie_duration['duration'].median(), inplace=True)

tv_show_duration = data[data['type'] == 'TV Show'].copy()
tv_show_duration['duration'] = tv_show_duration['duration'].astype(str).str.replace(' Seasons', '', regex=False).str.replace(' Season', '', regex=False)
tv_show_duration['duration'] = pd.to_numeric(tv_show_duration['duration'], errors='coerce')
tv_show_duration['duration'].fillna(tv_show_duration['duration'].median(), inplace=True)

data.loc[data['type'] == 'Movie', 'duration'] = movie_duration['duration']
data.loc[data['type'] == 'TV Show', 'duration'] = tv_show_duration['duration']
data['duration'] = data['duration'].astype(int)

# Load pre-computed matrices
try:
    with open('/content/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    with open('/content/cosine_sim.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)
except FileNotFoundError:
    st.error("Pre-computed matrices not found. Please run the notebook cells to generate them.")
    st.stop() # Stop the app if files are not found


def get_recommendations(title, cosine_sim=cosine_sim):
    """
    Gets content-based recommendations for a given movie title.

    Args:
        title (str): The title of the movie for which to get recommendations.
        cosine_sim (numpy.ndarray): The cosine similarity matrix.

    Returns:
        pandas.Series or str: A Series of the top 10 most similar movie titles
                                or a string message if the movie is not found.
    """
    indices = pd.Series(data.index, index=data['title']).drop_duplicates()
    if title not in indices.index:
        return "Movie title not found in the dataset."

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]
    return data['title'].iloc[movie_indices]

# Streamlit app interface
st.title('Netflix Movie Recommendation System')

# Selectbox for movie title
selected_movie = st.selectbox('Select a movie:', data['title'])

# Button to trigger recommendations
if st.button('Get Recommendations'):
    recommendations = get_recommendations(selected_movie)

    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.write(f"Recommendations for '{selected_movie}':")
        st.dataframe(recommendations)
