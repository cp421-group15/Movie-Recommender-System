from surprise import KNNWithMeans
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


print("KNN With Means:")
# SVD = Singular Value Decomposition
algorithm_UB = KNNWithMeans(sim_options={'name': 'pearson_baseline', 'user_based': True}, verbose=False)
algorithm_IB = KNNWithMeans(sim_options={'name': 'pearson_baseline', 'user_based': False}, verbose=False)

print("User-Based Collaborative Filtering Results:")
# Run 5-fold cross-validation and print results
cross_validate(algorithm_UB, movie_data, measures=['RMSE', 'MAE'], cv=10, verbose=True)

print("\nItem-Based Collaborative Filtering Results:")
# Run 5-fold cross-validation and print results
cross_validate(algorithm_IB, movie_data, measures=['RMSE', 'MAE'], cv=10, verbose=True)