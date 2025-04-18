import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from collections import defaultdict

# Helper function to determine age bin (from Code 2)
def get_age_bin(age):
    if age < 18:
        return "0-17"
    elif age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 55:
        return "45-54"
    elif age < 65:
        return "55-64"
    else:
        return "65+"

# Load and cache data with error handling
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
        
        # Precompute KNN model
        processed_users, _, _ = preprocess_data(users)
        features = processed_users.drop('user_id', axis=1).values
        knn_model = NearestNeighbors(n_neighbors=20, metric='cosine').fit(features)
        
        return users, ratings, movies, knn_model
    except FileNotFoundError as e:
        st.error(f"Critical data missing: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Enhanced preprocessing with feature weights
def preprocess_data(users):
    occupation_encoder = OneHotEncoder(sparse_output=False)
    occupation_encoded = occupation_encoder.fit_transform(users[['occupation']])
    occupation_df = pd.DataFrame(occupation_encoded, 
                               columns=occupation_encoder.get_feature_names_out(['occupation']))
    
    scaler = MinMaxScaler()
    users['age_scaled'] = scaler.fit_transform(users[['age']]) * 0.3  # Age weight
    
    # Gender encoding with reduced weight
    users['gender_enc'] = users['gender'].map({'M': 0, 'F': 1}) * 0.2
    
    # Occupation features with increased weight
    occupation_df = occupation_df * 0.5
    
    processed_users = pd.concat([users[['user_id', 'age_scaled', 'gender_enc']], occupation_df], axis=1)
    return processed_users, scaler, occupation_encoder

# Initialize session state with improved structure
def init_session_state():
    session_vars = {
        'movie_stats': defaultdict(lambda: {'impressions': 0, 'total_rating': 0, 'avg_rating': 0}),
        'total_recommendations': 0,
        'candidate_movies': {},
        'current_recommendations': [],
        'metrics': {
            'all_feedbacks': [],
            'diversity_scores': [],
            'precision_scores': [],
            'recall_scores': [],
            'coverage_scores': []
        }
    }
    for key, value in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Improved Reinforcement Learner with configurable parameters
class ReinforcementLearner:
    def __init__(self, exploration_factor=0.3, base_exploration=5.0):
        self.exploration_factor = exploration_factor
        self.base_exploration = base_exploration
        
    def calculate_ucb(self, movie_id):
        stats = st.session_state.movie_stats[movie_id]
        if stats['impressions'] == 0:
            return self.base_exploration + self.exploration_factor
        
        exploration = self.exploration_factor * np.sqrt(
            np.log(max(1, st.session_state.total_recommendations)) / 
            (stats['impressions'] + 1e-8))
        return stats['avg_rating'] + exploration

    def update_stats(self, movie_ids, rating):
        for movie_id in movie_ids:
            stats = st.session_state.movie_stats[movie_id]
            stats['impressions'] += 1
            stats['total_rating'] += rating
            stats['avg_rating'] = stats['total_rating'] / (stats['impressions'] + 1e-8)
        st.session_state.total_recommendations += len(movie_ids)

# Enhanced diversity calculation using entropy
def calculate_diversity(movies, recommended_ids):
    if not recommended_ids or len(recommended_ids) < 2:
        return 0
        
    genre_columns = movies.columns[5:24]
    recommended_movies = movies[movies['movie_id'].isin(recommended_ids)]
    genre_dist = recommended_movies[genre_columns].mean()
    
    entropy = -np.sum(genre_dist * np.log2(genre_dist + 1e-8))
    max_entropy = np.log2(len(genre_columns))
    return (entropy / max_entropy) if max_entropy > 0 else 0

def calculate_coverage(all_movies, recommended_movies):
    return len(recommended_movies) / len(all_movies)

def main():
    st.title("🎬 Movie Recommender Pro+")
    st.markdown("### Advanced Hybrid Recommendation System")
    
    # Load data with precomputed KNN
    users, ratings, movies, precomputed_knn = load_data()
    _, scaler, occupation_encoder = preprocess_data(users)
    
    # Initialize demographic totals (from Code 2)
    if 'age_totals' not in st.session_state:
        genres = movies.columns[5:24].tolist()
        age_bins = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        occupations = users['occupation'].unique().tolist()
        genders = ["M", "F"]
        
        st.session_state.age_totals = {bin_: {genre: {'total_rating': 0, 'count': 0} for genre in genres} for bin_ in age_bins}
        st.session_state.occupation_totals = {occ: {genre: {'total_rating': 0, 'count': 0} for genre in genres} for occ in occupations}
        st.session_state.gender_totals = {gen: {genre: {'total_rating': 0, 'count': 0} for genre in genres} for gen in genders}
        st.session_state.history = []
    
    # Configuration sidebar
    with st.sidebar.expander("⚙️ System Configuration"):
        exploration_factor = st.slider("Exploration Factor", 0.0, 1.0, 0.3)
        base_exploration = st.slider("Base Exploration Value", 1.0, 10.0, 5.0)
        cf_weight = st.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.7)
        rl_learner = ReinforcementLearner(
            exploration_factor=exploration_factor,
            base_exploration=base_exploration
        )
    
    # User input section
    with st.expander("📝 Enter Your Details", expanded=True):
        with st.form("user_info"):
            age = st.number_input("Age", min_value=1, max_value=100, value=25)
            gender = st.selectbox("Gender", ["M", "F"])
            occupation = st.selectbox("Occupation", users['occupation'].unique())
            submitted = st.form_submit_button("🚀 Get Recommendations")

    if submitted:
        # Store user demographics in session state (for use in feedback)
        st.session_state.user_age = age
        st.session_state.user_gender = gender
        st.session_state.user_occupation = occupation
        
        # Process user input
        new_user_occupation = occupation_encoder.transform([[occupation]])
        new_user_age = scaler.transform([[age]])
        gender_enc = 0 if gender == 'M' else 1
        
        # Create weighted feature vector
        user_features = np.concatenate([
            new_user_age.flatten(),
            [gender_enc],
            new_user_occupation.flatten()
        ]).reshape(1, -1)
        
        # Find similar users using precomputed KNN
        distances, indices = precomputed_knn.kneighbors(user_features)
        similar_users = users.iloc[indices[0]]['user_id']
        
        # Generate candidate movies
        similar_ratings = ratings[ratings['user_id'].isin(similar_users)]
        movie_ratings = similar_ratings.groupby('movie_id')['rating'].agg(['mean', 'count'])
        movie_ratings = movie_ratings[movie_ratings['count'] > 5]
        
        candidate_movies = movie_ratings.merge(movies[['movie_id', 'title']], on='movie_id')
        candidate_movies = candidate_movies.sort_values(by='mean', ascending=False).head(50)
        st.session_state.candidate_movies = candidate_movies.set_index('movie_id').to_dict('index')

    # Display recommendations
    if st.session_state.candidate_movies:
        st.markdown("---")
        st.markdown("## 🎥 Your Personal Recommendations")
        
        # Calculate scores
        movie_scores = []
        for movie_id, movie_data in st.session_state.candidate_movies.items():
            cf_score = movie_data['mean']
            ucb_score = rl_learner.calculate_ucb(movie_id)
            combined_score = (cf_weight * cf_score) + ((1 - cf_weight) * ucb_score)
            movie_scores.append((movie_id, combined_score, cf_score, ucb_score))
        
        # Get top 5 movies
        sorted_movies = sorted(movie_scores, key=lambda x: x[1], reverse=True)[:5]
        st.session_state.current_recommendations = [movie[0] for movie in sorted_movies]
        
        # Display recommendations
        cols = st.columns([3,2])
        with cols[0]:
            for idx, (movie_id, score, cf, ucb) in enumerate(sorted_movies):
                movie_title = st.session_state.candidate_movies[movie_id]['title']
                score_display = f"{score:.2f}" if ucb <= 100 else f"{cf:.2f}*"
                rl_display = f"{ucb:.2f}" if ucb <= 100 else "New! 🆕"
                
                st.markdown(f"""
                **{idx+1}. {movie_title}**  
                📊 Total Score: `{score_display}`  
                👥 Collaborative: `{cf:.2f}` | 🤖 RL Score: `{rl_display}`
                """)
        
        # Explanation panel
        with cols[1]:
            with st.expander("🔍 Recommendation Breakdown"):
                st.markdown("""
                **Algorithm Components:**
                - **Collaborative Filtering**: Based on similar users' ratings
                - **Reinforcement Learning**: Adapts based on system-wide feedback
                
                **Score Legend:**
                - *New! 🆕*: Never-before-recommended movie
                - *: Initial score without RL history
                """)
                st.latex(r'''UCB = \bar{X} + \alpha\sqrt{\frac{\ln N}{n + \epsilon}}''')

        # Feedback system
        st.markdown("---")
        with st.form("feedback_form"):
            st.markdown("### 💬 Rate Individual Movies")
            individual_ratings = {}
            
            for movie_id in st.session_state.current_recommendations:
                movie_title = st.session_state.candidate_movies[movie_id]['title']
                rating = st.slider(
                    f"Rate: {movie_title}",
                    1, 5, 3,
                    key=f"rate_{movie_id}"
                )
                individual_ratings[movie_id] = rating

            if st.form_submit_button("📤 Submit Ratings"):
                # Update RL model with individual ratings
                for movie_id, rating in individual_ratings.items():
                    rl_learner.update_stats([movie_id], rating)
                
                # Update metrics
                avg_feedback = np.mean(list(individual_ratings.values()))
                st.session_state.metrics['all_feedbacks'].append(avg_feedback)
                
                diversity = calculate_diversity(movies, st.session_state.current_recommendations)
                st.session_state.metrics['diversity_scores'].append(diversity)
                
                coverage = calculate_coverage(movies['movie_id'].unique(), 
                                            st.session_state.movie_stats.keys())
                st.session_state.metrics['coverage_scores'].append(coverage)
                
                # Update demographic totals (from Code 2)
                age_bin = get_age_bin(st.session_state.user_age)
                occupation = st.session_state.user_occupation
                gender = st.session_state.user_gender
                
                for movie_id, rating in individual_ratings.items():
                    movie_genres = movies[movies['movie_id'] == movie_id].iloc[0, 5:24]
                    genres_list = movie_genres[movie_genres == 1].index.tolist()
                    for genre in genres_list:
                        st.session_state.age_totals[age_bin][genre]['total_rating'] += rating
                        st.session_state.age_totals[age_bin][genre]['count'] += 1
                        st.session_state.occupation_totals[occupation][genre]['total_rating'] += rating
                        st.session_state.occupation_totals[occupation][genre]['count'] += 1
                        st.session_state.gender_totals[gender][genre]['total_rating'] += rating
                        st.session_state.gender_totals[gender][genre]['count'] += 1
                
                # Record historical averages (from Code 2)
                interaction = len(st.session_state.metrics['all_feedbacks'])
                for demographic_type, totals in zip(
                    ['Age Group', 'Occupation', 'Gender'],
                    [st.session_state.age_totals, st.session_state.occupation_totals, st.session_state.gender_totals]
                ):
                    for group, genre_totals in totals.items():
                        for genre, data in genre_totals.items():
                            avg_rating = data['total_rating'] / data['count'] if data['count'] > 0 else 0
                            st.session_state.history.append({
                                'interaction': interaction,
                                'demographic_type': demographic_type,
                                'group': group,
                                'genre': genre,
                                'avg_rating': avg_rating
                            })
                
                st.success("Feedback recorded! System updating... 🔄")

        # Performance dashboard with added Genre Preferences tab
        st.markdown("---")
        with st.expander("📊 System Performance Dashboard", expanded=True):
            tab1, tab2, tab3, tab4 = st.tabs(["User Feedback", "Recommendation Quality", "System Health", "Genre Preferences"])
            
            with tab1:
                if st.session_state.metrics['all_feedbacks']:
                    avg = np.mean(st.session_state.metrics['all_feedbacks'])
                    st.metric("Average Rating", f"{avg:.2f}/5")
                    fig = px.line(y=st.session_state.metrics['all_feedbacks'], 
                                title="User Satisfaction Trend")
                    st.plotly_chart(fig)
            
            with tab2:
                if st.session_state.metrics['diversity_scores']:
                    st.metric("Average Diversity", 
                            f"{np.mean(st.session_state.metrics['diversity_scores']):.2%}")
                    genre_counts = movies[movies['movie_id'].isin(st.session_state.current_recommendations)]\
                                  .iloc[:, 5:24].sum()
                    st.bar_chart(genre_counts)
            
            with tab3:
                st.metric("Catalog Coverage", 
                         f"{len(st.session_state.movie_stats)}/{len(movies)}",
                         f"{calculate_coverage(movies['movie_id'].unique(), 
                                            st.session_state.movie_stats.keys()):.2%}")
                st.metric("Average Recommendations/Movie", 
                         f"{np.mean([s['impressions'] for s in st.session_state.movie_stats.values()]):.1f}")
            
            with tab4:
                # Genre Preferences tab (from Code 2)
                if st.session_state.history:
                    history_df = pd.DataFrame(st.session_state.history)
                    st.subheader("Current Genre Preferences")
                    
                    demographic_type = st.selectbox("Demographic Type", ["Age Group", "Occupation", "Gender"])
                    if demographic_type == "Age Group":
                        groups = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
                        selected_group = st.selectbox("Age Group", groups)
                    elif demographic_type == "Occupation":
                        selected_group = st.selectbox("Occupation", users['occupation'].unique().tolist())
                    else:
                        selected_group = st.selectbox("Gender", ["M", "F"])
                    
                    # Get the latest interaction
                    max_interaction = history_df['interaction'].max()
                    filtered_df = history_df[(history_df['interaction'] == max_interaction) & 
                                             (history_df['demographic_type'] == demographic_type) & 
                                             (history_df['group'] == selected_group)]
                    
                    if not filtered_df.empty:
                        # Sort by avg_rating descending
                        filtered_df = filtered_df.sort_values('avg_rating', ascending=False)
                        fig = px.bar(filtered_df, x='genre', y='avg_rating',
                                    title=f"Current Genre Preferences for {demographic_type}: {selected_group}",
                                    labels={'avg_rating': 'Average Rating', 'genre': 'Genre'},
                                    text='avg_rating')
                        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                        st.plotly_chart(fig)
                    else:
                        st.write("No data yet for this selection.")
                else:
                    st.write("Submit ratings to see preferences.")

if __name__ == "__main__":
    init_session_state()
    main()