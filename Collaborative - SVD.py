import matplotlib.pyplot as plt
import pandas as pd

from surprise import accuracy
from surprise import SVD
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import PredefinedKFold, GridSearchCV, train_test_split, cross_validate

# Attributes in the data file are tab separated (\t).
# User = user_id
# Item = item_id
# Rating = rating
# Timestamp = timestamp
reader = Reader(line_format="user item rating timestamp", sep="\t", rating_scale=(1,5))

train_set = Dataset.load_from_file('./ml-100k/u1.base', reader=reader)

# Baseline SVD model
algorithm = SVD()

# Run 5-fold cross-validation and print results
baseline_results = cross_validate(algorithm, train_set, measures=['RMSE', 'MAE'], cv=5, verbose=True)

print("Baseline model - results on train set")
print(baseline_results)
print("------------------------")

# Tune the number of epochs and regularisation constant
print("Tuning number of epochs and regularisation constant")

param_grid = {'n_epochs': [40, 80],
              'reg_all': [0.1, 0.15, 0.2, 0.25, 0.5]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)

gs.fit(train_set)

print("Best params:")
print(gs.best_params)
print("Best scores:")
print(gs.best_score)
print("------------------------")

tuning_epoch_and_reg_results = gs.cv_results
tuning_epoch_and_reg_df = pd.DataFrame.from_dict(tuning_epoch_and_reg_results)
tuning_epoch_and_reg_df.to_csv("tuning_epoch_and_reg.csv", index=False)

fig = plt.figure()
ax = plt.axes()
plt.xlabel("regularisation constant")
plt.ylabel("mean validation rmse")

x1 = tuning_epoch_and_reg_df[tuning_epoch_and_reg_df["param_n_epochs"] == 40]["param_reg_all"]
y1 = tuning_epoch_and_reg_df[tuning_epoch_and_reg_df["param_n_epochs"] == 40]["mean_test_rmse"]
ax.plot(x1, y1)

x2 = tuning_epoch_and_reg_df[tuning_epoch_and_reg_df["param_n_epochs"] == 80]["param_reg_all"]
y2 = tuning_epoch_and_reg_df[tuning_epoch_and_reg_df["param_n_epochs"] == 80]["mean_test_rmse"]
ax.plot(x2, y2)

chartBox = ax.get_position()
ax.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])
ax.legend(["40", "80"], title="epochs", loc='upper center', bbox_to_anchor=(1.1, 0.8), shadow=True, ncol=1)
# plt.legend()
plt.show()

# Tune the number of factors
print("Tuning the number of factors")

param_grid = {'n_epochs': [80],
              'reg_all': [0.1],
              'n_factors': [120, 127, 135, 142, 150],
             }
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)

gs.fit(train_set)

print("Best params:")
print(gs.best_params)
print("Best scores:")
print(gs.best_score)
print("------------------------")

tuning_n_factors_results = gs.cv_results
tuning_n_factors_df = pd.DataFrame.from_dict(tuning_n_factors_results)
tuning_n_factors_df.to_csv("tuning_n_factors.csv", index=False)

fig = plt.figure()
ax = plt.axes()
plt.xlabel("n_factors")
plt.ylabel("mean validation rmse")

x1 = tuning_n_factors_df["param_n_factors"]
y1 = tuning_n_factors_df["mean_test_rmse"]
ax.plot(x1, y1)

plt.show()

# Tune the learning rate
print("Tuning the learning rate")

param_grid = {'n_epochs': [80],
              'reg_all': [0.1],
              'n_factors': [135],
              'lr_all': [0.001, 0.0025, 0.005, 0.01, 0.05]
             }
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=5)

gs.fit(train_set)

print("Best params:")
print(gs.best_params)
print("Best scores:")
print(gs.best_score)
print("------------------------")

tuning_lr_results = gs.cv_results
tuning_lr_results_df = pd.DataFrame.from_dict(tuning_lr_results)
tuning_lr_results_df.to_csv("tuning_lr_results.csv", index=False)

fig = plt.figure()
ax = plt.axes()
plt.xlabel("learning rate")
plt.ylabel("mean validation rmse")

x1 = tuning_lr_results_df["param_lr_all"]
y1 = tuning_lr_results_df["mean_test_rmse"]
ax.plot(x1, y1)

plt.show()

# Compare the baseline and tuned models on the test set
# Load test set
print("Running the baseline and tuned models on the test set")

folds_files = [("./ml-100k/u1.base", "./ml-100k/u1.test")]
data = Dataset.load_from_folds(folds_files, reader=reader)
pkf = PredefinedKFold()

test_set = None
for _, s in pkf.split(data):
    test_set = s
    
# Baseline model
print("Baseline model performance on test set")
print(accuracy.rmse(algorithm.test(test_set)))

# Tuned model
tuned_model = gs.best_estimator["rmse"]
tuned_model.fit(train_set.build_full_trainset())

print("Tuned model performance on test set")
print(accuracy.rmse(tuned_model.test(test_set)))