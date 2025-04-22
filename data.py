import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Preprocesses user data by encoding categorical features and scaling numerical features
def preprocess_data(users):
    # One-hot encode the 'occupation' column
    occupation_encoder = OneHotEncoder(sparse_output=False)
    occupation_encoded = occupation_encoder.fit_transform(users[['occupation']])
    occupation_df = pd.DataFrame(occupation_encoded, 
                               columns=occupation_encoder.get_feature_names_out(['occupation']))
    
    # Scale the 'age' column and encode the 'gender' column
    scaler = MinMaxScaler()
    users['age_scaled'] = scaler.fit_transform(users[['age']]) * 0.3
    users['gender_enc'] = users['gender'].map({'M': 0, 'F': 1}) * 0.2
    occupation_df = occupation_df * 0.5
    
    # Combine processed features into a single DataFrame
    processed_users = pd.concat([users[['user_id', 'age_scaled', 'gender_enc']], occupation_df], axis=1)
    return processed_users, scaler, occupation_encoder

# Caches the trained KNN model for efficient reuse
@st.cache_data
def train_knn(processed_users):
    # Extracts feature values and trains a KNN model
    features = processed_users.drop('user_id', axis=1).values
    return NearestNeighbors(n_neighbors=20, metric='cosine').fit(features)

# Loads and preprocesses data, handling errors gracefully
@st.cache_data
def load_data():
    try:
        # Load user, ratings, and movie data from files
        users = pd.read_csv('ml-100k/u.user', sep='|', encoding='latin-1', 
                           names=['user_id', 'age', 'gender', 'occupation', 'zip'])
        ratings = pd.read_csv('ml-100k/u.data', sep='\t', 
                            names=['user_id', 'movie_id', 'rating', 'timestamp'])
        movies = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1',
                            names=['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
                                   'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
        
        # Preprocess user data and train the KNN model
        processed_users, scaler, occupation_encoder = preprocess_data(users)
        knn_model = train_knn(processed_users)
        
        # Extract genre columns from the movies DataFrame
        genre_cols = movies.filter(regex='^(Action|Adventure|Animation|Children|Comedy|Crime|Documentary|Drama|Fantasy|Film-Noir|Horror|Musical|Mystery|Romance|Sci-Fi|Thriller|War|Western)$').columns
        return users, ratings, movies, knn_model, processed_users, scaler, occupation_encoder, genre_cols
    except FileNotFoundError as e:
        # Handle missing file errors
        st.error("Critical data missing. Please upload the required files.")
        st.stop()
    except Exception as e:
        # Handle other errors during data loading
        st.error(f"Error loading data: {str(e)}")
        st.stop()