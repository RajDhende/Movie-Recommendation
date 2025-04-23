# Import necessary libraries and modules
import streamlit as st
import numpy as np
import plotly.express as px
from collections import defaultdict
from data import load_data
from recommender import ReinforcementLearner, create_user_features, find_candidate_movies, score_and_select_movies, calculate_coverage

# Function to initialize session state variables
def init_session_state(movies, users):
    if 'movie_stats' not in st.session_state:
        # Dictionary to store movie statistics
        st.session_state.movie_stats = defaultdict(lambda: {'impressions': 0, 'total_rating': 0, 'avg_rating': 0})
    if 'total_recommendations' not in st.session_state:
        # Counter for total recommendations made
        st.session_state.total_recommendations = 0
    if 'candidate_movies' not in st.session_state:
        # Dictionary to store candidate movies for recommendation
        st.session_state.candidate_movies = {}
    if 'current_recommendations' not in st.session_state:
        # List to store current recommendations
        st.session_state.current_recommendations = []
    if 'metrics' not in st.session_state:
        # Dictionary to store system metrics
        st.session_state.metrics = {
            'all_feedbacks': [],
            'coverage_scores': []
        }
    if 'age_totals' not in st.session_state:
        # Initialize demographic-based statistics
        genres = movies.columns[5:24].tolist()
        age_bins = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
        occupations = users['occupation'].unique().tolist()
        genders = ["M", "F"]
        
        st.session_state.age_totals = {bin_: {genre: {'total_rating': 0, 'count': 0} for genre in genres} for bin_ in age_bins}
        st.session_state.occupation_totals = {occ: {genre: {'total_rating': 0, 'count': 0} for genre in genres} for occ in occupations}
        st.session_state.gender_totals = {gen: {genre: {'total_rating': 0, 'count': 0} for genre in genres} for gen in genders}
        st.session_state.history = []

# Function to render the user input form
def render_user_form(users):
    with st.expander("📝 Enter Your Details", expanded=True):
        with st.form("user_info"):
            # Input fields for user details
            age = st.number_input("Age", min_value=1, max_value=100, value=25)
            gender = st.selectbox("Gender", ["M", "F"])
            occupation = st.selectbox("Occupation", users['occupation'].unique())
            submitted = st.form_submit_button("🚀 Get Recommendations")
    return age, gender, occupation, submitted

# Function to display recommendations
def render_recommendations(sorted_movies):
    st.markdown("## 🎥 Your Personal Recommendations")
    cols = st.columns([3, 2])
    with cols[0]:
        for idx, (movie_id, score, cf, ucb) in enumerate(sorted_movies):
            # Display movie details and scores
            movie_title = st.session_state.candidate_movies[movie_id]['title']
            score_display = f"{score:.2f}" if ucb <= 100 else f"{cf:.2f}*"
            rl_display = "New! 🆕" if st.session_state.movie_stats[movie_id]['impressions'] == 0 else f"{ucb:.2f}"
            st.markdown(f"**{idx+1}. {movie_title}**  \n📊 Total Score: `{score_display}`  \n👥 Collaborative: `{cf:.2f}` | 🤖 RL Score: `{rl_display}`")

# Function to process user feedback and update system metrics
def process_feedback(individual_ratings, rl_learner, movies):
    # Update movie statistics based on feedback
    st.session_state.total_recommendations = rl_learner.update_stats(
        individual_ratings,  
        st.session_state.movie_stats,
        st.session_state.total_recommendations
    )
    
    # Calculate average feedback and coverage
    avg_feedback = np.mean(list(individual_ratings.values()))
    st.session_state.metrics['all_feedbacks'].append(avg_feedback)
    coverage = calculate_coverage(movies['movie_id'].unique(), st.session_state.movie_stats.keys())
    st.session_state.metrics['coverage_scores'].append(coverage)

# Main function to run the Streamlit app
def main():
    st.title("🎬 Movie Recommender Pro+")
    st.markdown("### Advanced Hybrid Recommendation System")
    
    # Load data and initialize session state
    users, ratings, movies, knn_model, processed_users, scaler, occupation_encoder, genre_cols = load_data()
    init_session_state(movies, users)
    
    # Sidebar configuration for system parameters
    with st.sidebar.expander("⚙️ System Configuration"):
        exploration_factor = st.slider("Exploration Factor", 0.0, 1.0, 0.3)
        base_exploration = st.slider("Base Exploration Value", 1.0, 10.0, 5.0)
        cf_weight = st.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.7)
        rl_learner = ReinforcementLearner(exploration_factor=exploration_factor, base_exploration=base_exploration)
    
    # Render user input form
    age, gender, occupation, submitted = render_user_form(users)

    if submitted:
        # Store user details in session state
        st.session_state.user_age = age
        st.session_state.user_gender = gender
        st.session_state.user_occupation = occupation
        
        # Generate user features and find candidate movies
        user_features = create_user_features(age, gender, occupation, scaler, occupation_encoder)
        candidate_movies = find_candidate_movies(user_features, knn_model, processed_users, ratings, movies)
        st.session_state.candidate_movies = candidate_movies

    if st.session_state.candidate_movies:
        st.markdown("---")
        # Score and select movies for recommendation
        sorted_movies = score_and_select_movies(
            st.session_state.candidate_movies,
            rl_learner,
            st.session_state.movie_stats,
            st.session_state.total_recommendations,
            cf_weight=cf_weight
        )
        st.session_state.current_recommendations = [movie[0] for movie in sorted_movies]
        
        # Render recommendations
        render_recommendations(sorted_movies)
        
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

        st.markdown("---")
        with st.form("feedback_form"):
            st.markdown("### 💬 Rate Individual Movies")
            individual_ratings = {}
            for movie_id in st.session_state.current_recommendations:
                # Collect user ratings for recommended movies
                movie_title = st.session_state.candidate_movies[movie_id]['title']
                rating = st.slider(f"Rate: {movie_title}", 1, 5, 3, key=f"rate_{movie_id}")
                individual_ratings[movie_id] = rating

            if st.form_submit_button("📤 Submit Ratings"):
                # Process feedback and update system
                process_feedback(individual_ratings, rl_learner, movies)
                st.success("Feedback recorded! System updating... 🔄")

        st.markdown("---")
        with st.expander("📊 System Performance Dashboard", expanded=True):
            tab1, tab2, tab3 = st.tabs(["User Feedback", "Recommendation Quality", "System Health"])
            
            with tab1:
                if st.session_state.metrics['all_feedbacks']:
                    # Display user feedback metrics
                    avg = np.mean(st.session_state.metrics['all_feedbacks'])
                    st.metric("Average Rating", f"{avg:.2f}/5")
                    fig = px.line(y=st.session_state.metrics['all_feedbacks'], title="User Satisfaction Trend")
                    st.plotly_chart(fig)
            
            with tab2:
                # Display genre distribution of recommendations
                genre_counts = movies[movies['movie_id'].isin(st.session_state.current_recommendations)].iloc[:, 5:24].sum()
                st.bar_chart(genre_counts)
            
            with tab3:
                # Display system health metrics
                st.metric("Catalog Coverage", 
                         f"{len(st.session_state.movie_stats)}/{len(movies)}",
                         f"{calculate_coverage(movies['movie_id'].unique(), st.session_state.movie_stats.keys()):.2%}")
                st.metric("Average Recommendations/Movie", 
                         f"{np.mean([s['impressions'] for s in st.session_state.movie_stats.values()]):.1f}")

if __name__ == "__main__":
    main()