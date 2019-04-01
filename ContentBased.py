import pandas as pd
import pprint
import tabulate
from sklearn.metrics.pairwise import cosine_similarity

# Import preprocesed IMDB movie metadata.
IMDB_data = pd.read_csv('./IMDB Data/processed_data_final.csv')

def lower_case(var):
    return str.lower(var)

IMDB_data['Movie_Title'] = IMDB_data['Movie_Title'].apply(lower_case)

# To normalize strings in the data.
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

# Normalize the String data.
for feature in ['Director', 'Actors', 'Plot_Keywords', 'Studios', 'Genres']:
    IMDB_data[feature] = IMDB_data[feature].apply(normalize_data)

# To make description.
def make_description(data):
    description = ''
    for feature in ['Director', 'Actors', 'Plot_Keywords', 'Studios', 'Genres']:
        description += ''.join(data[feature])
    return description

# Make description.
IMDB_data['Description'] = IMDB_data.apply(make_description, axis=1)

# Create a CountVectorizer.
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')

# Get the term-document matrix.
td_matrix = count_vectorizer.fit_transform(IMDB_data['Description'])

# Reset index.
IMDB_data = IMDB_data.reset_index()

# Create a list of data indices & Movie Titles.
indices = pd.Series(IMDB_data.index, index=IMDB_data['Movie_Title'])

# Function to get a recommendation.
def recommend(movie_title):

    # Lower case the query.
    movie_title = str.lower(movie_title)
    
    # Create a cosine similarity object.
    c_similarity = cosine_similarity(td_matrix, td_matrix)

    # Get the index of the movie.
    index = indices[movie_title]

    # Get the cosine similarity for all movies.
    cosine_similarities = list(enumerate(c_similarity[index]))

    # Sort the movies based on the cosine similarities.
    cosine_similarities = sorted(cosine_similarities, key=lambda x: x[1], reverse=True)

    # Get the top 10 similarities.
    cosine_similarities = cosine_similarities[1:11]

    recommended_indices = []

    # Get the movie indices
    for i in range(0, 10):
        recommended_indices.append((cosine_similarities[i][0]))

    # Return the top 10 most similar movies
    movie_titles = IMDB_data['Movie_Title'].iloc[recommended_indices]

    recommendations = []
    k = 0
    for i, row in movie_titles.iteritems():
        recommendations.append([i, movie_titles[i], cosine_similarities[k][1]])
        k+=1 

    # Returns items in format: [index, title, cosine_similarity]
    return recommendations

exists = False

while exists == False:
    try:
        movie_name = input("Enter a movie in format 'name (YYYY)':\n")
        print("")

        # Compute the cosine similarity matrix
        recommended = recommend(movie_name)

        prints = []
        for i in range(0, 5):
            prints.append([recommended[i][1], recommended[i][2]])

        print(tabulate.tabulate(prints, headers=['Movie Title', 'Cosine Similarity']))
        exists = True
    except:
        print("Movie does not exist in the data set, try again:")
    
