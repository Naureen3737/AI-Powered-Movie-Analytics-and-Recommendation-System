###----------------------------------------------------------------------------------------
import pickle
import numpy as np
import re

# -------------------------------
# 1) Load trained model
# -------------------------------
with open(r"C:\Users\HP\Desktop\Movie_Project\app\model\svd_model.pkl", "rb") as f:
    model = pickle.load(f)

predicted_ratings = model['predicted_ratings']
user_ids = model['user_ids']
movie_ids = model['movie_ids']
movie_df = model['movie_df']

# -------------------------------
# 2) Extract year helper
# -------------------------------
def extract_year(title):
    match = re.search(r'\((\d{4})\)', title)
    return match.group(1) if match else 'N/A'

def recommend_movies(user_id, top_n=5):
    if user_id not in user_ids:
        # New user → fallback to top-rated movies
        top_movies_all = movie_df.groupby('movieId')['rating'].mean().sort_values(ascending=False)
        top_movies_all = top_movies_all.index.tolist()
    else:
        user_index = user_ids.index(user_id)
        user_pred_ratings = predicted_ratings[user_index, :]
        # Sort by predicted rating descendingg
        top_indices = np.argsort(user_pred_ratings)[::-1]
        top_movies_all = []
        for idx in top_indices:
            movie_id = movie_ids[idx]
            # Skip movies already rated by user
            if movie_df[(movie_df['userId_rating'] == user_id) & 
                        (movie_df['movieId'] == movie_id)].shape[0] == 0:
                top_movies_all.append(movie_id)

    # ---------------------------------------------------
    # ✔ Recommend 5 movies with NO repeating genres
    # ---------------------------------------------------
    selected_movies = []
    used_genres = set()

    for movie_id in top_movies_all:
        row = movie_df[movie_df['movieId'] == movie_id].iloc[0]
        movie_genres = str(row['genres']).split('|')

        # If ANY genre of this movie already used → skip
        if any(g in used_genres for g in movie_genres):
            continue

        # Otherwise recommend this movie
        selected_movies.append({
            "title": row['title'],
            "year": extract_year(row['title']),
            "genres": row['genres']
        })

        # Mark ALL its genres as used
        used_genres.update(movie_genres)

        # Stop after 5 movies
        if len(selected_movies) >= top_n:
            break

    return selected_movies







