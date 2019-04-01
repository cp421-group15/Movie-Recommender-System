import os
import sys
import random
import numpy as np
import pandas as pd
import pprint
import json

# Dictionaries to store useful information.
DIRECTORS = dict()
STUDIOS = dict()
ACTORS = dict()
MOVIES = dict()
ASPECT_RATIOS = dict()

# Features to drop from IMDB dataset.
IMDB_DROPS = ['color', 'num_critic_for_reviews', 'director_facebook_likes',
              'actor_3_facebook_likes', 'actor_2_name', 'actor_1_facebook_likes',
              'actor_1_name', 'num_voted_users', 'cast_total_facebook_likes',
              'actor_3_name', 'facenumber_in_poster', 'movie_imdb_link',
              'num_user_for_reviews', 'actor_2_facebook_likes', 'imdb_score',
              'movie_facebook_likes']

# Features to drop from TMDB dataset.
TMDB_DROPS = ['budget', 'genres', 'homepage', 'keywords', 'original_language',
              'overview', 'popularity', 'production_countries', 'runtime', 'spoken_languages',
              'status', 'tagline', 'original_title']

# ---
# Import Movie Datasets.
# ---

# Import IMDB dataset.
imdb_movie_data = pd.read_csv(os.path.dirname(os.path.abspath(
    __file__))+"\\imdb-5000-movie-dataset\\movie_metadata.csv")

# Import TMDB dataset.
tmdb_movie_data = pd.read_csv(os.path.dirname(os.path.abspath(
    __file__))+"\\tmdb-5000-movie-dataset\\tmdb_5000_movies.csv")

# ---
# IMDB Processing.
# ---

# Adjust IMDB name.
for i, row in imdb_movie_data.iterrows():
    title = row['movie_title']
    try:
        imdb_movie_data.at[i, 'movie_title'] = title.strip()
    except Exception as e:
        pass

# ---
# TMDB Processing.
# ---

tmdb_movie_data.rename(columns={'title': 'movie_title'}, inplace=True)
tmdb_movie_data['title_year'] = tmdb_movie_data['release_date'].copy()

# Adjust Year
for i, row in tmdb_movie_data.iterrows():
    date = row['title_year']
    try:
        tmdb_movie_data.at[i, 'title_year'] = np.float64(date.split('-')[0])
    except Exception as e:
        pass

# Adjust TMDB name.
for i, row in tmdb_movie_data.iterrows():
    title = row['movie_title']
    try:
        tmdb_movie_data.at[i, 'movie_title'] = title.strip()
    except Exception as e:
        pass

# Change the release year to numeric.
tmdb_movie_data['title_year'] = pd.to_numeric(tmdb_movie_data['title_year'])

# ---
# Dropping Columns.
# ---

# Drop columns that we don't care about.
imdb_movie_data = imdb_movie_data.drop(columns=IMDB_DROPS)
tmdb_movie_data = tmdb_movie_data.drop(columns=TMDB_DROPS)

# ---
# Merging the Datasets.
# ---

# Merge the data.
full_data = pd.merge(imdb_movie_data, tmdb_movie_data,
                     how='inner', on=['movie_title', 'title_year'])
# Drop duplicates.
full_data = full_data.drop_duplicates()

# Remove columns where country of production is not USA.
#full_data = full_data.loc[full_data['country'] == 'USA']

# Add year to movie title.
for i, row in full_data.iterrows():
    movie_title = row['movie_title']+" ("+str(int(row['title_year']))+")"
    # Release date in format YYYY-MM-DD
    full_data.loc[full_data['movie_title']==row['movie_title'], 'movie_title'] = movie_title

# Release year no longer needed. Was only needed for join.
full_data = full_data.drop(columns=['title_year', 'country'])

# --
# Normalize Names Thus Far.
# --

full_data = full_data.rename(columns={'director_name': 'Director', 'duration': 'Runtime', 'gross': 'Gross', 'genres': 'Genres', 'movie_title': 'Movie_Title',
                                      'plot_keywords': 'Plot_Keywords', 'language': 'Language', 'content_rating': 'Content_Rating', 'budget': 'Budget',
                                      'aspect_ratio': 'Aspect_Ratio', 'id': 'Movie_ID', 'production_companies': 'Production_Companies',
                                      'release_date': 'Release_Date', 'revenue': 'Revenue', 'vote_average':'Vote_Average', 'vote_count':'Vote_Count'})


# --
# Import Credits Dataset.
# --

tmdb_credit_data = pd.read_csv(os.path.dirname(os.path.abspath(
    __file__))+"\\tmdb-5000-movie-dataset\\tmdb_5000_credits.csv")


#full_data.to_csv(os.path.dirname(os.path.abspath(__file__))+"\\data_before_processing.csv", index=False)

# --
# Informative data processing..
# --

# Replace missing budgets with the mean budget across all films remaining.
full_data['Budget'] = full_data['Budget'].replace(np.nan, np.float64(0))
full_data['Budget'] = full_data['Budget'].replace(np.float64(
    0), np.mean(pd.to_numeric(full_data['Budget'][full_data['Budget'] != np.float64(0)], errors='throw')))


# ---
# ACTOR DETAILS. --> Get top 3 actors for each film.
# ---

# Create new columns for actors.
full_data['Actors'] = ""

# Get top three actors for each movie.
for i, row in full_data.iterrows():
    # Get the id of the movie.
    tmdb_movie_id = row['Movie_ID']
    # Find the credits for that movie.
    cast = tmdb_credit_data.loc[tmdb_credit_data['movie_id'] == tmdb_movie_id]['cast'].item()

    # Set Dataframe Info.
    try:
        cast_json = json.loads(cast)
        actor_id = cast_json[0]['id']
        full_data.at[i, 'Actors'] = [cast_json[0]['name'],cast_json[1]['name'],cast_json[2]['name']]

    except Exception as e:
        pass

# ---
# DIRECTOR DETAILS. --> Director Name.
# ---
# Note: Director name is already present.
"""

# Create new columns for actors.
full_data['Director'] = None

# Get top three actors for each movie.
for i, row in full_data.iterrows():
    # Get the id of the movie.
    tmdb_movie_id = row['Movie_ID']
    # Find the credits for that movie.
    crew = tmdb_credit_data.loc[tmdb_credit_data['movie_id'] == tmdb_movie_id]['crew'].item()

    # Set Dataframe Info.
    try:
        crew_json = json.loads(crew)
        # Find director details in crew list.
        location = 0
        for j, crew_member in enumerate(crew_json):
            if crew_member['job'].lower() == 'director':
                location = j
                break
        # Reference director details.
        director_details = crew_json[location]
        full_data.at[i, 'Director'] = director_details['name']

    except Exception as e:
        pass

"""

# ---
# STUDIO DETAILS. --> Studio_IDs, Studio_Names
# Note: 4710 studios.
# ---

# Create new columns for actors.
full_data['Studios'] = np.empty((len(full_data), 0)).tolist()

for i, row in full_data.iterrows():
    # Get the id of the movie.
    tmdb_movie_id = row['Movie_ID']

    studio_ids = []
    studio_names = []
    for studio in json.loads(row['Production_Companies']):
        studio_id = studio['id']
        studio_ids.append(studio_id)
        studio_names.append(studio['name'])

    if len(studio_names) < 3:
        full_data.at[i, 'Studios'] = studio_names[0:len(studio_names)]
    else:
        full_data.at[i, 'Studios'] = studio_names[0:3]

# --
# Useful data processing..
# --

# --
# Genres Dataset. One Hot Encoding.
# --
GENRES = set()
FULL_GENRES = dict()
genre_count=0

# Create a list of all existing genres in the data set.
for i, row in full_data.iterrows():
    genres = row['Genres'].split('|')
    for genre in genres:
        GENRES.add(genre)

    genre_list = row['Genres']

    if genre_list not in FULL_GENRES.keys():
        FULL_GENRES[genre_list] = genre_count
        genre_count += 1

# Create new columns for all genres, each of format: Genre_Comedy.
# The default value of each column will be 0.
for genre in GENRES:
    full_data['Genre_'+genre] = np.float64(0)

full_data['Full_Genre'] = np.float64(0)

# Assign 1 to the columns for each movie that has a given genre.
for i, row in full_data.iterrows():
    genres = row['Genres'].split('|')
    for genre in genres:
        full_data.at[i, "Genre_"+genre] = np.float64(1)
    full_data.at[i, 'Full_Genre'] = np.float(FULL_GENRES[row['Genres']])


# --
# - Release_Month
#    --> Ordinal (Label Encoding --> 1 to 12)
# --

# Create column for the release month.
full_data['Release_Month'] = np.float64(0)

for i, row in full_data.iterrows():
    release_date = row['Release_Date']
    # Release date in format YYYY-MM-DD
    release_month = np.float64(release_date.split('-')[1]) - 1
    # Set the release month for the movie.
    full_data.at[i, 'Release_Month'] = release_month

# --
#- Language
#    --> ONLY English
# Note: 4211 of the films are English.
# --
full_data = full_data[full_data['Language']=='English']

# Uncomment the below line to remove all
# full_data = full_data.loc[full_data['Country'] == 'USA'])

# --
# - Content_Rating_Score
#    --> Ordinal. (Label Encoding)
# --

full_data['Content_Rating_Score'] = np.float64(0)

# Content Ratings on a scale of maturity level.
rating_scale = {'G': 0,
                'TV-G': 0,
                'GP': 0,
                'Approved': 0,
                'Passed': 0,
                'PG': 1,
                'TV-PG': 1,
                'PG-13': 2,
                'TV-14': 2,
                'R': 3,
                'M': 3,
                'Unrated': 3,
                'Not Rated': 3,
                'NC-17': 4,
                'X': 4,
                'Disapproved': 4
                }

for i, row in full_data.iterrows():
    try:
        content_rating = row['Content_Rating']
        # If content rating is null, assume G.
        if not isinstance(content_rating, str) and np.isnan(content_rating):
            content_rating = 'G'
        # Set rating value based on scale.
        rating_score = rating_scale[content_rating]
        full_data.at[i, 'Content_Rating_Score'] = np.float64(rating_score)
    except Exception as e:
        pass

# --
#- Budget
#    --> Continuous.
# --

DROP_FINAL = ['Gross', 'Production_Companies', 'Language']
full_data = full_data.drop(columns=DROP_FINAL)

# Replace meissing runtimes with median runtime.
full_data['Runtime'] = full_data['Runtime'].replace(np.nan, np.float64(0))
full_data['Runtime'] = full_data['Runtime'].replace(np.float64(
    0), np.median(pd.to_numeric(full_data['Runtime'][full_data['Runtime'] != np.float64(0)], errors='throw')))
print(np.mean(pd.to_numeric(full_data['Runtime'][full_data['Runtime'] != np.float64(0)], errors='throw')))

# Replace aspect ratio with median aspect ratio.
full_data['Aspect_Ratio'] = full_data['Aspect_Ratio'].replace(np.nan, np.float64(0))
full_data['Aspect_Ratio'] = full_data['Aspect_Ratio'].replace(np.float64(
    0), np.median(pd.to_numeric(full_data['Aspect_Ratio'][full_data['Aspect_Ratio'] != np.float64(0)], errors='throw')))
print(np.median(pd.to_numeric(full_data['Aspect_Ratio'][full_data['Aspect_Ratio'] != np.float64(0)], errors='throw')))


for i, row in full_data.iterrows():
    full_data.at[i, "Genres"] = list(str(row["Genres"]).split("|"))
    full_data.at[i, "Plot_Keywords"] = list(str(row["Plot_Keywords"]).split("|"))

# Proven unuseful.
full_data = full_data.drop(columns=['Genre_History', 'Genre_Mystery', 'Genre_Sport', 'Genre_Film-Noir',
       'Genre_Thriller', 'Genre_Action', 'Genre_Drama', 'Genre_Horror',
       'Genre_Short', 'Genre_Family', 'Genre_Crime', 'Genre_Sci-Fi',
       'Genre_War', 'Genre_Animation', 'Genre_Romance', 'Genre_News',
       'Genre_Musical', 'Genre_Comedy', 'Genre_Fantasy', 'Genre_Western',
       'Genre_Documentary', 'Genre_Music', 'Genre_Adventure',
       'Genre_Biography', 'Full_Genre', 'Release_Month'])

print(full_data.columns)

full_data.to_csv(os.path.dirname(os.path.abspath(__file__))+"\\processed_data_final.csv", index=False)

