import numpy as np
import pandas as pd
import random
from collections import defaultdict


# Step 2: Define custom hash functions
def generate_hash_functions(num_hashes, max_rows):
    hash_functions = []
    for i in range(1, num_hashes + 1):
        a, b = np.random.randint(1, max_rows, size=2)  # Random coefficients
        hash_functions.append(lambda x, a=a, b=b: (a * x + b) % max_rows)
    return hash_functions

def createBinaryMatrix(df):

    users = df['userId'].unique()
    movies = df['movieId'].unique()

    user_to_index = {user: i for i, user in enumerate(users)}
    movie_to_index = {movie: i for i, movie in enumerate(movies)}

    num_users = len(users)
    num_movies = len(movies)

    # Initialize the characteristic matrix
    characteristic_matrix = np.zeros((num_movies, num_users), dtype=int)

    # Fill the characteristic matrix
    for _, row in df.iterrows():
        user_idx = user_to_index[row['userId']]
        movie_idx = movie_to_index[row['movieId']]
        characteristic_matrix[movie_idx, user_idx] = 1

    return characteristic_matrix



def MinHashFunction(num_hashes, characteristic_matrix, m):
    """
    Optimized MinHash function using row groups and partial computation.

    Args:
    - num_users: Number of users (columns).
    - num_hashes: Number of hash functions to use.
    - characteristic_matrix: Binary characteristic matrix (rows: items, cols: users).
    - m: Number of rows to process per group.

    Returns:
    - signature_matrix: A matrix of MinHash signatures.
    """

    num_movies = characteristic_matrix.shape[0]
    num_users = characteristic_matrix.shape[1]
    k = num_movies
    num_groups = k // m
    # Initialize the signature matrix with infinity
    signature_matrix = np.full((num_hashes * num_groups, num_users), np.inf)

    # Create random hash functions
    hash_functions = [
        (lambda x, a=a, b=b, p=k: (a * x + b) % p)
        for a, b in zip(
            random.sample(range(1, k), num_hashes),
            random.sample(range(0, k), num_hashes),
        )
    ]

    for group in range(num_groups):
        start_row = group * m
        end_row = start_row + m

        # Process only the rows in the current group
        group_matrix = characteristic_matrix[start_row:end_row, :]

        # Iterate through hash functions
        for hash_idx, h in enumerate(hash_functions):
            # Compute hash values for the rows in the group
            row_hashes = np.array([h(row) for row in range(start_row, end_row)])

            # Update the signature matrix for each column (user)
            for user in range(num_users):
                # Skip columns with all 0s in this group
                user_column = group_matrix[:, user]
                if np.any(user_column == 1):  # Only process columns with 1's
                    min_hash = np.min(row_hashes[user_column == 1])
                    signature_idx = group * num_hashes + hash_idx
                    signature_matrix[signature_idx, user] = min(
                        signature_matrix[signature_idx, user], min_hash
                    )

    return signature_matrix



def lsh_user_clustering(signature_matrix, num_bands = 3):
    """
    Apply LSH to cluster similar users based on their MinHash signatures.
    Args:
        signature_matrix: MinHash signature matrix with shape (num_hashes, num_users).
        num_bands: Number of bands to divide the signature matrix into.
    Returns:
        A dictionary where keys are buckets (hash values), and values are lists of user indices.
    """

    num_hashes, num_users = signature_matrix.shape
    rows_per_band = num_hashes // num_bands

    buckets = defaultdict(list)

    # Hash users within each band
    for band in range(num_bands):
        start_row = band * rows_per_band
        end_row = start_row + rows_per_band
        band_matrix = signature_matrix[start_row:end_row, :] 

        # Hash each user's signature in this band
        for user_index in range(num_users):
            band_signature = tuple(band_matrix[:, user_index])
            buckets[band_signature].append(user_index)

    return buckets


def query_similar_users(user_index, buckets, signature_matrix, num_candidates=2):
    """
    Identify the most similar users for a given user.
    Args:
        user_index: The index of the target user.
        buckets: The LSH buckets from lsh_user_clustering.
        signature_matrix: MinHash signature matrix.
        num_candidates: Number of most similar users to return.
    Returns:
        A list of indices of the most similar users.
    """
    candidate_users = set()

    # Identify all buckets the user is part of
    for bucket_users in buckets.values():
        if user_index in bucket_users:
            candidate_users.update(bucket_users)

    # Remove the target user from candidates
    candidate_users.discard(user_index)

    # Compute Jaccard similarity for candidate users
    user_signature = signature_matrix[:, user_index]
    similarities = []
    for candidate in candidate_users:
        candidate_signature = signature_matrix[:, candidate]
        # Jaccard Similarity
        similarity = np.mean(user_signature == candidate_signature) 
        similarities.append((candidate, similarity))

    # Sort candidates by similarity and return top ones
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [user for user, _ in similarities[:num_candidates]]


def recommend_movies(similar_users, user_movie_ratings, max_recommendations=5):
    """
    Recommend movies for a user based on similar users' ratings.
    Args:
        user_index: Index of the target user.
        similar_users: List of indices of the most similar users.
        user_movie_ratings: DataFrame with columns ['userId', 'movieId', 'title', 'rating'].
        max_recommendations: Maximum number of movies to recommend.
    Returns:
        A list of recommended movies and their average ratings.
    """
    # Filter ratings for the target user and similar users
    similar_users_ratings = user_movie_ratings[user_movie_ratings['userId'].isin(similar_users)]
    # List of 
    # Find common movies rated by similar users
    recommendations = []
    for movie_id, group in similar_users_ratings.groupby('movieId'):
        if len(group['userId'].unique()) > 1:  # At least two similar users rated the movie
            avg_rating = group['rating'].mean()
            title = group['title'].iloc[0]
            recommendations.append((movie_id, title, avg_rating))

    # Sort recommendations by average rating
    recommendations.sort(key=lambda x: x[2], reverse=True)

    # If recommendations are insufficient, add top-rated movies from the most similar user
    if len(recommendations) < max_recommendations:
        most_similar_user = similar_users[0]
        top_movies = similar_users_ratings[similar_users_ratings['userId'] == most_similar_user]
        top_movies = top_movies.sort_values(by='rating', ascending=False)
        for _, row in top_movies.iterrows():
            if len(recommendations) >= max_recommendations:
                break
            if row['movieId'] not in [rec[0] for rec in recommendations]:
                recommendations.append((row['movieId'], row['title'], row['rating']))

    recommended_movies = pd.DataFrame(data = [[movie[1], movie[2]] for movie in recommendations[:max_recommendations]], columns = ["Movies Recommended","Rating"])
    # Return at most max_recommendations
    return recommended_movies
