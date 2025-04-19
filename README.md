# 🎬 Movie Recommender Pro+

**Movie Recommender Pro+** is an advanced hybrid movie recommendation system built with **Streamlit**, combining **Collaborative Filtering**, **Reinforcement Learning**, and **demographic-based analysis** to deliver highly personalized and adaptive recommendations. The system not only suggests movies based on similar users but also dynamically adjusts using real-time user feedback to improve its recommendations over time.

---

## 📌 Features

- 🔍 **Hybrid Recommendation Engine**  
  Combines **Collaborative Filtering (KNN-based)** with a **Reinforcement Learning Upper Confidence Bound (UCB) model** to balance known favorites with new discoveries.

- 📊 **Dynamic Feedback Integration**  
  Users rate recommended movies, and the system instantly updates its statistics and adapts future suggestions based on feedback.

- 📈 **Performance & Insights Dashboard**
  - Track **user satisfaction trends**
  - Visualize **genre diversity** and **catalog coverage**
  - Monitor **demographic-specific genre preferences** (by Age, Gender, and Occupation)

- 🎛️ **Configurable Parameters**
  - Adjust **Exploration Factor**, **Base Exploration Value**, and **Collaborative Filtering Weight** on the fly through a sidebar configuration panel.

- 📚 **Demographic-Aware Genre Trends**
  - Track genre preferences for different demographic groups
  - Visualize trends dynamically based on real-time feedback and interactions

---

## 📂 Project Structure

```
.
├── ml-100k/
│   ├── u.user                # User demographic data
│   ├── u.data                # Movie ratings data
│   └── u.item                # Movie metadata
├── recommender_app.py        # Main Streamlit application (this file)
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies

Make sure you have Python 3.8+ installed, then run:

```bash
pip install streamlit pandas numpy scikit-learn plotly
```

### 2️⃣ Download MovieLens Dataset

Download the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/)  
Unzip it and place the `ml-100k` folder in the project directory.

### 3️⃣ Run the App

```bash
streamlit run recommender_app.py
```

---

## 🧠 How It Works

### 📚 Collaborative Filtering (KNN)
Finds users with similar demographic profiles and movie preferences using **cosine similarity** in a weighted feature space (age, gender, occupation).

### 🤖 Reinforcement Learning (UCB)
Uses the **Upper Confidence Bound** formula to balance:
- **Exploitation**: Movies known to perform well
- **Exploration**: New or lesser-seen movies with high potential

### 🎨 Diversity & Coverage
Calculates:
- **Diversity** using **genre entropy**
- **Coverage** based on the proportion of unique movies recommended out of the full catalog

### 📈 Demographic-Based Trends
Tracks and visualizes genre preferences segmented by:
- **Age Group**
- **Gender**
- **Occupation**

---

## 📊 Dashboard Preview

- **User Feedback Trend:** 📈  
  Track average ratings over time.

- **Recommendation Quality:** 🎥  
  Measure diversity of genres in recommendations.

- **System Health:** 💾  
  Monitor catalog coverage and average impressions per movie.

- **Genre Preferences:** 🎨  
  Real-time demographic insights into favorite genres.

---

## ✨ Future Improvements

- Add **content-based filtering** using movie metadata
- Support **movie posters** and **trailers**
- Deploy on **Streamlit Cloud** or **Dockerized deployment**
- Add **user authentication** and persistent profiles
- Enhance exploration strategies (e.g., Thompson Sampling, Epsilon-Greedy)



---

