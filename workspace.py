import argparse
from pandas import read_csv
import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
from tqdm import tqdm

from logger import logger


def RMSE(Z, T):
    R = (Z - T) ** 2
    return (np.nansum(R) / np.count_nonzero(~np.isnan(R))) ** 0.5

def ParseArguments():
    parser = argparse.ArgumentParser(description="Recommendation System")
    parser.add_argument('--train', default="train_ratings.csv", required=False,
                        help='file with csv file containing training data (default: %(default)s)')
    parser.add_argument('--test', default="test_ratings.csv", required=False,
                        help='file with csv file containing testing data (default: %(default)s)')
    parser.add_argument('--alg', default="SVD2", required=False,
                        help='algorithm that you want to use (default: %(default)s)')
    parser.add_argument('--result', default="", required=False,
                        help='file where a final RMSE will be saved (default: %(default)s)')
    parser.add_argument('--r', default=2, required=False, type=int,
                        help=' ')
    parser.add_argument('--fill_method', default="user_specific", required=False,
                        help='file where a final RMSE will be saved (default: %(default)s)')

    args = parser.parse_args()

    return args.train, args.test, args.alg, args.result, args.r, args.fill_method

training_file, testing_file, alg, output_file, r, fill_method = ParseArguments()

training = read_csv(training_file).drop(columns=['timestamp'])
test = read_csv(testing_file).drop(columns=['timestamp'])

## reshape to matrix

test = test.pivot_table(index='userId', columns='movieId', values='rating')


training = training.pivot_table(index='userId', columns='movieId', values='rating')

test = test.reindex(columns=training.columns)

def fill_with_original_values(orginal_data, edited_data):
    return np.where(np.isnan(orginal_data), edited_data, orginal_data)

def fill_with_col_mean(data):
    col_means = data.mean(skipna=True, axis=0)
    return data.fillna(col_means)

def fill_with_col_mean(data):
    col_median = data.median(skipna=True, axis=0)
    return data.fillna(col_median)

def fill_with_row_mean(data):
    row_means = data.mean(skipna=True, axis=1)
    return data.transpose().fillna(row_means).transpose()

def fill_with_row_median(data):
    row_median = data.median(skipna=True, axis=1)
    return data.transpose().fillna(row_median).transpose()

def fill_with_arbitrary_number(data, number: float):
    return data.fillna(number)

def fill_with_user_specific(data, alpha: float = 0.4):
    return alpha * fill_with_col_mean(data) + (1 - alpha) * fill_with_row_mean(data)

def round_and_clip(data):
    return np.clip((np.round(data * 2) / 2),1, 5)

match fill_method: #TO DO
    case 'movie_mean':
        pass
    case 'user_mean':
        pass
    case 'user_median':
        pass
    case 'movie_median':
        pass
    case 'arbitrary_number':
        pass
    case 'user_specific':
        pass


#alpha = 0.4 # wydaje sie dobrze dzialac dodac argparse

def NMF_my(data, r = 20):
    model = NMF(n_components=r, init='random', random_state=0, max_iter=1000)
    W = model.fit_transform(data)
    H = model.components_
    return np.dot(W, H)

def SVD(data, r=16):
    svd = TruncatedSVD(n_components=r, random_state=42)
    svd.fit(data)
    Sigma2 = np.diag(svd.singular_values_)
    VT = svd.components_
    W = svd.transform(data) / svd.singular_values_
    H = np.dot(Sigma2, VT)
    return np.dot(W, H)

def SVD_iterated(data, r_: int=2, max_iter: int=40):
    Z = SVD(data, r_)
    for _ in tqdm(range(max_iter-1)):
        Z = fill_with_original_values(orginal_data=training, edited_data=Z)
        Z = np.clip(Z, 1, 5) # czy clipowac?
        tuple = (r_, 0.4, _, RMSE(round_and_clip(Z), test.values))
        tuple = str(tuple).strip('()')
        logger.info(tuple)
        Z = SVD(Z, r_)
    return Z

logger.info('r, alpha, iter, RMSE')

Z_combined = fill_with_user_specific(training, 0.4)
SVD_iterated(Z_combined, r, 50)

# Z_approximated = SVD_iterated(Z_combined, r, 10)
# training_matrix = training.values
# training_matrix = fill_with_original_values(training, Z_approximated)
# training_matrix = round_and_clip(training_matrix)
# RMSE, accuracy = my_rmse(training_matrix, test.values)
# print(r, RMSE, accuracy)
