import pandas as pd
import pprint

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

'''
['Director', 'Runtime', 'Genres', 'Movie_Title', 'Plot_Keywords',
    'Content_Rating', 'Budget', 'Aspect_Ratio', 'Movie_ID',
    'Release_Date_x', 'Revenue', 'Vote_Average', 'Vote_Count', 'Actors',
    'Studios', 'Content_Rating_Score', 'Movie_Lens_ID', 'Release_Date_y',
    'Video_Release_Date', 'IMDB_URL']
'''

'''
Similarity based on:
1. Director
2. Runtime
3. Genres
4. Plot Keywords
5. Content Rating
6. Vote Average (Or Content Rating Score)
7. Actors
8. Studios
'''

drops = ['Content_Rating', 'Budget', 'Aspect_Ratio', 'Movie_ID',
       'Release_Date_x', 'Revenue', 'Vote_Count', 'Content_Rating_Score',
       'Movie_Lens_ID', 'Release_Date_y', 'Video_Release_Date', 'IMDB_URL']

common_movies_data = common_movies_data.drop(columns=drops)

print(common_movies_data.columns)

# Calculate TF-IDF for keywords?
'''
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(common_movies_data['Plot_Keywords'])

# 1233 plot keywords used to describe 338 films.
print(tfidf_matrix.shape)
'''

# Clean the data, by removing spaces.


def normalize_data(var):
    # If is a list.
    if isinstance(var, list):
        item_list = var
        for i, item in enumerate(item_list):
            item_list[i] = str.lower(item_list[i].replace(" ", ""))
        return item_list
    # If is a string.
    elif isinstance(var, str):
        return str.lower(var.replace(" ", ""))
    # Otherwise.
    else:
        return ""


for feature in ['Director', 'Actors', 'Plot_Keywords', 'Studios', 'Genres']:
    common_movies_data[feature] = common_movies_data[feature].apply(
        normalize_data)

# DF with exactly what we want to use.
# ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


def create_soup(x):
    return ' '.join(x['Director']) + ' ' + ' '.join(x['Actors']) + ' ' + x['Plot_Keywords'] + ' ' + ' '.join(x['Studios']) + ' ' + ' '.join(x['Genres'])

common_movies_data['soup'] = common_movies_data.apply(create_soup, axis=1)

from sklearn.feature_extraction.text import CountVectorizer

count=CountVectorizer(stop_words='english')
count_matrix=count.fit_transform(common_movies_data['soup'])

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

common_movies_data = common_movies_data.reset_index()

indices = pd.Series(common_movies_data.index, index=common_movies_data['Movie_Title'])


# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Use Count Vectorizer on the director, actor.
def get_recommendations(title, cosine_sim=cosine_sim2):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    print(movie_indices)

    # Return the top 10 most similar movies
    return common_movies_data['Movie_Title'].iloc[movie_indices]

print(get_recommendations(common_movies_data['Movie_Title'].iloc[8]))

