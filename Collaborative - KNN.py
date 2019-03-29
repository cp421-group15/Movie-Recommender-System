from surprise import KNNBasic
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import cross_validate

'''
http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
'''

# Attributes in the data file are tab separated (\t).
# User = user_id
# Item = item_id
# Rating = rating
# Timestamp = timestamp
reader = Reader(line_format="user item rating timestamp", sep="\t", rating_scale=(1,5))
movie_data = Dataset.load_from_file('./ml-100k/u.data', reader=reader)

# SVD = Singular Value Decomposition
algorithm = KNNBasic()

# Run 5-fold cross-validation and print results
cross_validate(algorithm, movie_data, measures=['RMSE', 'MAE'], cv=10, verbose=True)