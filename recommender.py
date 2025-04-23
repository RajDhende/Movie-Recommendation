import numpy as np
import pandas as pd

class ReinforcementLearner:
    def __init__(self, exploration_factor=0.3, base_exploration=5.0):
        # Initializes the reinforcement learner with exploration parameters
        self.exploration_factor = exploration_factor
        self.base_exploration = base_exploration
        
    def calculate_ucb(self, movie_id, movie_stats, total_recommendations):
        # Calculates the Upper Confidence Bound (UCB) score for a given movie
        stats = movie_stats[movie_id]
        if stats['impressions'] == 0:
            return self.base_exploration + self.exploration_factor
        exploration = self.exploration_factor * np.sqrt(np.log(max(1, total_recommendations)) / (stats['impressions'] + 1e-8))
        return stats['avg_rating'] + exploration

    def update_stats(self, movie_ratings, movie_stats, total_recommendations):
        # Updates movie statistics based on new ratings
        for movie_id, rating in movie_ratings.items():
            stats = movie_stats[movie_id]
            stats['impressions'] += 1
            stats['total_rating'] += rating
            stats['avg_rating'] = stats['total_rating'] / (stats['impressions'] + 1e-8)
        return total_recommendations + len(movie_ratings)

def create_user_features(age, gender, occupation, scaler, occupation_encoder):
    # Creates a feature vector for a user based on their demographic data
    occupation_df = pd.DataFrame([[occupation]], columns=['occupation'])
    age_df = pd.DataFrame([[age]], columns=['age'])

    occupation_encoded = occupation_encoder.transform(occupation_df)
    age_scaled = scaler.transform(age_df)

    gender_enc = 0 if gender == 'M' else 1

    user_features = np.concatenate([age_scaled.flatten(), [gender_enc], occupation_encoded.flatten()]).reshape(1, -1)
    return user_features

def find_candidate_movies(user_features, knn_model, processed_users, ratings, movies, min_ratings=5, top_n=50):
    # Finds candidate movies for recommendation based on similar users
    distances, indices = knn_model.kneighbors(user_features)
    similar_users = processed_users.iloc[indices[0]]['user_id']
    similar_ratings = ratings[ratings['user_id'].isin(similar_users)]
    movie_ratings = similar_ratings.groupby('movie_id')['rating'].agg(['mean', 'count'])
    movie_ratings = movie_ratings[movie_ratings['count'] > min_ratings]
    candidate_movies = movie_ratings.merge(movies[['movie_id', 'title']], on='movie_id')
    candidate_movies = candidate_movies.sort_values(by='mean', ascending=False).head(top_n)
    return candidate_movies.set_index('movie_id').to_dict('index')

def score_and_select_movies(candidate_movies, rl_learner, movie_stats, total_recommendations, cf_weight=0.7, top_k=5):
    # Scores candidate movies using collaborative filtering and UCB, then selects the top movies
    movie_scores = []
    for movie_id, movie_data in candidate_movies.items():
        cf_score = movie_data['mean']
        ucb_score = rl_learner.calculate_ucb(movie_id, movie_stats, total_recommendations)
        combined_score = (cf_weight * cf_score) + ((1 - cf_weight) * ucb_score)
        movie_scores.append((movie_id, combined_score, cf_score, ucb_score))
    sorted_movies = sorted(movie_scores, key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_movies

def calculate_coverage(all_movies, recommended_movies):
    # Calculates the coverage of recommendations as a fraction of all available movies
    return len(recommended_movies) / len(all_movies)