import pandas as pd
# For TF-IDF.
from sklearn.feature_extraction.text import TfidfVectorizer

# Import preprocessed IMDB movie metadata.
metadata = pd.read_csv('./IMDB Data/processed_data_final.csv')

# Import Movie Lens Movies data set.
movie_lens_movie_data = pd.read_csv('ml-100k/u.item', sep='|', names=['Movie_Lens_ID', 'Movie_Title', 'Release_Date', 'Video_Release_Date', 'IMDB_URL'], usecols=range(5),
                     encoding='latin-1')

# Only keep movies which are common between the two data sets.
common_movies_data = pd.merge(metadata, movie_lens_movie_data,
                     how='inner', on=['Movie_Title'])

# Import user ratings data set.
user_ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['User_ID', 'Movie_Lens_ID', 'Rating', 'Timestamp'],
                      encoding='latin-1')

# 338 Common IDs.
valid_IDs = set(common_movies_data['Movie_Lens_ID'].unique())

# Remove ratings for movies not in the common data set.
# Reduces dataset from size 100,000 to 41,421
user_ratings = user_ratings[user_ratings['Movie_Lens_ID'].isin(valid_IDs)]


# Calculate Weighted Average based on IMDB formula. MAYBE.



# Calculate TF-IDF for keywords?


# Clean the data, by removing spaces?


# DF with exactly what we want to use.


# Use Count Vectorizer on the director, actor.