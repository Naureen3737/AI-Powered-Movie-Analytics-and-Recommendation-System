import pandas as pd
import numpy as np
import pickle

# -------------------------------
# 1) Load Dataset
# -------------------------------
df = pd.read_excel(r"C:\Users\HP\Desktop\Movie_Project\data\Movie_Detail_filled.xlsx")
df.columns = df.columns.str.strip()

# Only need userId, movieId, rating
ratings = df[['userId_rating', 'movieId', 'rating']].copy()
ratings['userId_rating'] = ratings['userId_rating'].astype(int)

# Create user-movie rating matrix
rating_matrix = ratings.pivot_table(
    index='userId_rating',
    columns='movieId',
    values='rating'
).fillna(0)

R = rating_matrix.to_numpy()
num_users, num_movies = R.shape

# -------------------------------
# 2) SVD Matrix Factorization
# -------------------------------
K = 20         # latent factors
steps = 50     # iterations
alpha = 0.002  # learning rate
beta = 0.02    # regularization

# Initialize latent matrices
P = np.random.rand(num_users, K)
Q = np.random.rand(num_movies, K)

# Gradient Descent
for step in range(steps):
    for i in range(num_users):
        for j in range(num_movies):
            if R[i, j] > 0:
                eij = R[i, j] - np.dot(P[i, :], Q[j, :].T)
                P[i, :] += alpha * (2 * eij * Q[j, :] - beta * P[i, :])
                Q[j, :] += alpha * (2 * eij * P[i, :] - beta * Q[j, :])
    if step % 10 == 0:
        error = 0
        for i in range(num_users):
            for j in range(num_movies):
                if R[i, j] > 0:
                    error += (R[i, j] - np.dot(P[i, :], Q[j, :].T))**2
                    error += (beta/2) * (np.sum(P[i, :]**2) + np.sum(Q[j, :]**2))
        print(f"Iteration {step}, error = {error:.4f}")

# Predicted ratings
predicted_ratings = np.dot(P, Q.T)

# -------------------------------
# 3) Save Model
# -------------------------------
with open("svd_model.pkl", "wb") as f:
    pickle.dump({
        "predicted_ratings": predicted_ratings,
        "user_ids": rating_matrix.index.tolist(),
        "movie_ids": rating_matrix.columns.tolist(),
        "movie_df": df
    }, f)

print("Training complete. Model saved as svd_model.pkl")
