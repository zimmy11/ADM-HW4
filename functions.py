import numpy as np
import pandas as pd
import random
from collections import defaultdict
import re

def createBinaryMatrix(df):
    """
    Function to create the Binary Matrix from a DataFrame df

    Args:
    - df: DataFrame containing all the rating given to all the movies present 

    Returns:
    - characteristic_matrix: A Sparse matrix of 1 and 0 if the userId has rated the movieId
    """
    # Get unique user IDs from the DataFrame
    users = df['userId'].unique()

    # Get unique movie IDs from the DataFrame
    movies = df['movieId'].unique()

    # Create a mapping from user ID to index
    user_to_index = {user: i for i, user in enumerate(users)}

    # Create a mapping from movie ID to index
    movie_to_index = {movie: i for i, movie in enumerate(movies)}

    # Count the total number of unique users
    num_users = len(users)

    # Count the total number of unique movies
    num_movies = len(movies)

    # Initialize a characteristic matrix with dimensions [num_movies x num_users]
    # Each cell will be set to 0 initially, representing no interaction
    characteristic_matrix = np.zeros((num_movies, num_users), dtype=int)

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Map the user ID in the row to its corresponding index
        user_idx = user_to_index[row['userId']]
        
        # Map the movie ID in the row to its corresponding index
        movie_idx = movie_to_index[row['movieId']]
        
        # Set the corresponding cell in the characteristic matrix to 1
        # This indicates that the user has interacted with the movie
        characteristic_matrix[movie_idx, user_idx] = 1

    # Return or use the characteristic matrix as needed
    return characteristic_matrix




def MinHashFunction(num_hashes, characteristic_matrix, m):
    """
    Optimized MinHash function using row groups and partial computation.

    Args:
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



def lsh_user_movies(signature_matrix, num_bands = 3):
    """
    Apply LSH to cluster similar users based on their MinHash signatures.
    Args:
        signature_matrix: MinHash signature matrix with shape (num_hashes, num_users).
        num_bands: Number of bands to divide the signature matrix into.
    Returns:
        A dictionary where keys are buckets (hash values), and values are lists of user indices.
    """

    # Get the dimensions of the signature matrix
    num_hashes_movies, num_users = signature_matrix.shape

    # Calculate how many rows each band should contain
    rows_per_band = num_hashes_movies // num_bands

    buckets = defaultdict(list)

    # Hash users within each band
    for band in range(num_bands):
        
        # Determine the range of rows corresponding to the current band
        start_row = band * rows_per_band
        end_row = start_row + rows_per_band
        # Extract the submatrix (rows for this band) from the signature matrix
        band_matrix = signature_matrix[start_row:end_row, :] 

        # Hash each user's signature in this band
        for user_index in range(num_users):
            # Create a tuple representing the band signature for this user
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
        similar_users: List of indices of the most similar users.
        user_movie_ratings: DataFrame with columns ['userId', 'movieId', 'title', 'rating'].
        max_recommendations: Maximum number of movies to recommend.
    Returns:
        A list of recommended movies and their average ratings.
    """
    # Filter ratings for the target user and similar users
    similar_users_ratings = user_movie_ratings[user_movie_ratings['userId'].isin(similar_users)]

    recommendations = []
    # Loop over the similar_users_rating grouped by MovieId
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

        # Filter the ratings of the most similar user from the 'similar_users_ratings' DataFrame
        top_movies = similar_users_ratings[similar_users_ratings['userId'] == most_similar_user]

        # Sort the filtered movies by their ratings in descending order
        # This ensures that the highest-rated movies by the most similar user are considered first
        top_movies = top_movies.sort_values(by='rating', ascending=False)
        for _, row in top_movies.iterrows():
            # Stop adding recommendations if the desired number of recommendations is reached
            if len(recommendations) >= max_recommendations:
                break
            if row['movieId'] not in [rec[0] for rec in recommendations]:
                recommendations.append((row['movieId'], row['title'], row['rating']))

    # We build a DataFrame extracting the Movie's Name, and its Avergae rating limiting for max_recommendation
    recommended_movies = pd.DataFrame(data = [[movie[1], movie[2]] for movie in recommendations[:max_recommendations]], columns = ["Movies Recommended","Rating"])

    return recommended_movies


# Function to clean and reorder the dataset by removing unnecessary columns
def new_order(dataset_movies_ratings):
    """
    This function takes in a dataset containing movie ratings, removes unnecessary columns 
    ('userId' and 'timestamp'), and reorders the columns to ['movieId', 'title', 'genres', 'rating'].
    
    Args:
        dataset_movies_ratings: DataFrame containing movie ratings with columns ['userId', 'movieId', 'title', 'genres', 'rating', 'timestamp'].
    
    Returns:
        DataFrame: A cleaned dataset with only the relevant columns ['movieId', 'title', 'genres', 'rating'].
    """
    # Drop 'userId' and 'timestamp' columns
    new_dataset = dataset_movies_ratings.drop(columns=['userId','timestamp'])  
    #  Define the desired column order
    order = ['movieId', 'title', 'genres', 'rating']  
    # Reorder the columns
    new_dataset = new_dataset[order]  
    return new_dataset

# Function to calculate the average rating for each movie
def average_ra(new_dataset):
    """
    This function calculates the average rating for each movie in the dataset.
    It removes empty strings and movies with only spaces in the 'genres' column and calculates the average rating.
    
    Args:
        new_dataset: DataFrame containing columns ['movieId', 'title', 'genres', 'rating'].
    
    Returns:
        DataFrame: The cleaned dataset with an additional column 'ratings_avg' containing the average rating for each movie.
    """
    # Remove the empty strings and movies with just spaces in the 'genres' column
    new_dataset = new_dataset[new_dataset['genres'].apply(lambda x: isinstance(x, str) and bool(x.strip()))]
    # Group by movie title and calculate the mean rating
    ratings_avg = new_dataset.groupby('title')['rating'].mean()  
    # Drop the 'rating' column
    new_dataset = new_dataset.drop(columns=['rating'])  
    # Round the average ratings to 1 decimal place
    arrotondo = [round(num, 1) for num in ratings_avg.values]  
    # Remove duplicate movie titles
    new_dataset = new_dataset.drop_duplicates(subset='title', keep='first')  
    # Add the average ratings to the dataset
    new_dataset['ratings_avg'] = arrotondo  
    return new_dataset

# Function to determine a representative genre for a movie based on its genre tags
def representative_genre(genres):
    """
    This function assigns a representative genre to a movie based on the genres provided.
    The genre selection is based on certain combinations of genres.
    
    Args:
        genres: String containing the movie genres, separated by '|'.
    
    Returns:
        str: The representative genre of the movie.
    """
     # Convert genres to lowercase
    genres = genres.lower() 
     # Split the genres into a list
    genres = genres.split("|") 
    if 'children' in genres:
        return 'children'
    if 'fantasy' in genres and 'mystery' in genres:
        return 'mystery'
    if 'action' in genres and 'adventure' in genres:
        return 'action adventure'
    if 'action' in genres and 'fantasy' in genres:
        return 'fantasy action'
    if 'thriller' in genres and 'horror' in genres:
        return 'horror'
    if 'biographical' in genres:
        return 'biographical'
    if 'historical' in genres:
        return 'historical'
    if 'comedy' in genres and 'drama' in genres:
        return 'comedy'  
    # Default genre if no specific combination is found
    return genres[0]

# Function to read the genome tags dataset
def read_gen_tag():
    """
    This function reads the 'genome_tags.csv' file and removes rows where the 'tag' is empty or just spaces.
    
    Returns:
        DataFrame: The cleaned dataset with genome tags.
    """
    movie_tag = "archive/genome_tags.csv"
    dataset_genome_tags = pd.read_csv(movie_tag)
    # Remove the empty tags
    dataset_genome_tags = dataset_genome_tags[dataset_genome_tags['tag'].apply(lambda x: isinstance(x, str) and bool(x.strip()))]
    return dataset_genome_tags

# Function to read the genome scores dataset
def read_gen_score():
    """
    This function reads the 'genome_scores.csv' file and removes rows with missing values.
    
    Returns:
        DataFrame: The cleaned dataset with genome scores.
    """
    genome_score = "archive/genome_scores.csv"
    dataset_genome_score = pd.read_csv(genome_score)
    # Remove rows with missing values
    dataset_genome_score = dataset_genome_score.dropna()
    return dataset_genome_score

# Function to add the most relevant genome tag to the movie dataset
def relevant_genome(new_dataset, dataset_tag):
    """
    This function merges the dataset with the most relevant genome tag for each movie.
    The relevance is determined by the highest 'relevance' score for each movie.
    
    Args:
        new_dataset: DataFrame containing movie data (including 'movieId').
        dataset_tag: DataFrame containing the genome tag and relevance score for each movie.
    
    Returns:
        DataFrame: The dataset with an additional 'relevant_genome_tag' column.
    """
    # Filter dataset_tag to the tag with the highest relevance for each movie
    idx_max_relevance = dataset_tag.groupby('movieId')['relevance'].idxmax()
    df_max_relevance = dataset_tag.loc[idx_max_relevance]
    # Merge the relevant genome tag with the main dataset
    new_dataset = pd.merge(new_dataset, df_max_relevance, on='movieId', how='inner')
    # Drop unnecessary columns
    new_dataset = new_dataset.drop(columns=['tagId', 'relevance'])  
    # Rename the tag column
    new_dataset = new_dataset.rename(columns={'tag': 'relevant_genome_tag'})  
    return new_dataset

# Function to read the common tags used by users in the 'tag.csv' file
def read_common():
    """
    This function reads the 'tag.csv' file, removes rows with empty tags, and drops the 'timestamp' column.
    
    Returns:
        DataFrame: The cleaned dataset containing user-assigned tags for movies.
    """
    tag = "archive/tag.csv"
    person_tag = pd.read_csv(tag)
    # Remove the empty tags
    person_tag = person_tag[person_tag['tag'].apply(lambda x: isinstance(x, str) and bool(x.strip()))]
     # Drop 'timestamp' column
    person_tag = person_tag.drop(columns=['timestamp']) 
    return person_tag

# Function to get the most frequently occurring tags for each movie
def get_max_occuring_tags(group):
    """
    This function identifies the most frequently occurring tag(s) for each movie based on the count.
    
    Args:
        group: A DataFrame containing movieId and tag count for each tag associated with the movie.
    
    Returns:
        list: A list of tagIds that have the highest frequency for the movie.
    """
    # Find the maximum occurrence count
    max_count = group['count'].max()   
    # Find the tags with the maximum count
    max_tags = group[group['count'] == max_count]['tagId'].tolist() 
    return max_tags

# Function to get the most common tag for each movie based on user tags
def common_column(person_tag):
    """
    This function calculates the most common tag for each movie based on user-assigned tags.
    
    Args:
        person_tag: DataFrame containing columns ['movieId', 'tag', 'count'] for user tags.
    
    Returns:
        DataFrame: A DataFrame with the most common tag for each movie.
    """
    # Count the occurrences of each tag for each movie
    tag_counts = person_tag.groupby(['movieId', 'tag']).size().reset_index(name='count')
    # Find the most common tag for each movie
    max_tag_per_movie = tag_counts.loc[tag_counts.groupby('movieId')['count'].idxmax()]
    tag_counts_single_tag = tag_counts.merge(max_tag_per_movie[['movieId', 'tag']], on='movieId', how='left', suffixes=('', '_max'))
    # Drop duplicate tags
    common_user_tag = tag_counts_single_tag[['movieId', 'tag_max']].drop_duplicates()  
    return common_user_tag

# Function to extract numbers (like year) from movie title
def extract_numbers_from_title(title):
    """
    This function extracts all the numbers enclosed in parentheses from the movie title.
    
    Args:
        title: A string containing the movie title.
    
    Returns:
        list: A list of numbers (e.g., years) found in the title.
    """
    # Find numbers inside parentheses
    numbers = re.findall(r'\((\d+)\)', title)  
    return numbers

# Function to remove numbers (like years) from the movie title
def remove_numbers_from_title(title):
    """
    This function removes numbers enclosed in parentheses and other digits from the movie title.
    
    Args:
        title: A string containing the movie title.
    
    Returns:
        str: The cleaned title without numbers.
    """
    # Remove numbers inside parentheses
    clean_title = re.sub(r'\(\d+\)', '', title)  
    # Remove all digits
    clean_title = re.sub(r'\d+', '', clean_title)  
    return clean_title