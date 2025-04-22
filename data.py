import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def preprocess_data(users):
    occupation_encoder = OneHotEncoder(sparse_output=False)
    occupation_encoded = occupation_encoder.fit_transform(users[['occupation']])
    occupation_df = pd.DataFrame(occupation_encoded, 
                               columns=occupation_encoder.get_feature_names_out(['occupation']))
    
    scaler = MinMaxScaler()
    users['age_scaled'] = scaler.fit_transform(users[['age']]) * 0.3
    users['gender_enc'] = users['gender'].map({'M': 0, 'F': 1}) * 0.2
    occupation_df = occupation_df * 0.5
    
    processed_users = pd.concat([users[['user_id', 'age_scaled', 'gender_enc']], occupation_df], axis=1)
    return processed_users, scaler, occupation_encoder

@st.cache_data
def load_data():
    try:
        users = pd.read_csv('ml-100k/u.user', sep='|', encoding='latin-1', 
                           names=['user_id', 'age', 'gender', 'occupation', 'zip'])
        ratings = pd.read_csv('ml-100k/u.data', sep='\t', 
                            names=['user_id', 'movie_id', 'rating', 'timestamp'])
        movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                            names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
                                   'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
        
        processed_users, scaler, occupation_encoder = preprocess_data(users)
        features = processed_users.drop('user_id', axis=1).values
        knn_model = NearestNeighbors(n_neighbors=20, metric='cosine').fit(features)
        
        return users, ratings, movies, knn_model, processed_users, scaler, occupation_encoder
    except FileNotFoundError as e:
        st.error(f"Critical data missing: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()