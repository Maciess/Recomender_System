import argparse
from pandas import read_csv, DataFrame
import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn import preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import logging
import sys


def RMSE(Z, T):
    R = (Z - T) ** 2
    return (np.nansum(R) / np.count_nonzero(~np.isnan(R))) ** 0.5

def ParseArguments():
    parser = argparse.ArgumentParser(description="Recommendation System")
    parser.add_argument('--train', default="train_ratings.csv", required=False,
                        help='file with csv file containing training data (default: %(default)s)')
    parser.add_argument('--test', default="test_ratings.csv", required=False,
                        help='file with csv file containing testing data (default: %(default)s)')
    parser.add_argument('--ALG', default="SGD", required=False,
                        help='algorithm that you want to use (default: %(default)s)')
    parser.add_argument('--result', default="result", required=False,
                        help='file where a final RMSE will be saved (default: %(default)s)')
    parser.add_argument('--r', default=2, required=False, type=int,
                        help=' ')
    parser.add_argument('--fill_method', default="user_specific", required=False,
                        help='file where a final RMSE will be saved (default: %(default)s)')

    args = parser.parse_args()

    return args.train, args.test, args.ALG, args.result, args.r, args.fill_method

training_file, testing_file, alg, output_file, r, fill_method = ParseArguments()


#logging mechanism
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler(output_file, mode= 'w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)


logger.addHandler(file_handler)
logger.addHandler(stdout_handler)




training = read_csv(training_file).drop(columns=['timestamp'])
test = read_csv(testing_file).drop(columns=['timestamp'])

train_index = training[['userId', 'movieId']].values
train_rating = training['rating'].values
test_index = test[['userId', 'movieId']].values
test_rating = test['rating'].values

## reshape to matrix


training = training.pivot_table(index='userId', columns='movieId', values='rating')

test = test.pivot_table(index='userId', columns='movieId', values='rating')



test = test.reindex(columns=training.columns)






def fill_with_original_values(orginal_data, edited_data):
    return np.where(np.isnan(orginal_data), edited_data, orginal_data)

def fill_with_col_mean(data):
    col_means = data.mean(skipna=True, axis=0)
    return data.fillna(col_means)

def fill_with_col_median(data):
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
    return np.clip((np.round(data * 2) / 2), 1, 5)

match fill_method:
    case 'movie_mean':
        Z  = fill_with_col_mean(training)
    case 'user_mean':
        Z  = fill_with_row_mean(training)
    case 'user_median':
        Z  = fill_with_row_median(training)
    case 'movie_median':
        Z  = fill_with_col_median(training)
    case 'arbitrary_number':
        Z  = fill_with_arbitrary_number(training, number=3)
    case 'user_specific':
        Z  = fill_with_user_specific(training, alpha=.4)


def NMF_my(data, r = 25):
    model = NMF(n_components=r, init='random', random_state=0, max_iter=300)
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
    for _ in range(max_iter-1):
        Z = fill_with_original_values(orginal_data=training, edited_data=Z)
        Z = np.clip(Z, 1, 5)
        Z = SVD(Z, r_)
    return Z

def SGD(train_index, test_index, train_rating, test_rating):
    X_train, X_test, y_train, y_test = train_index, test_index, train_rating, test_rating

    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = DataFrame(data=X_train)
    X_train['rating'] = list(y_train)
    X_test = DataFrame(data=X_test)
    X_test['rating'] = list(y_test)
    sgd_model = SGDRegressor(penalty='l2', alpha=10,power_t=0.45, eta0=0.9)
    sgd_model.fit(X_train.values, y_train)
    y_pred = sgd_model.predict(X_test.values)
    y_pred = np.clip((np.round(y_pred * 2) / 2), 1, 5)

    return y_pred

def sgd_rmse(Z, T):
    T_vals = []
    i = 0
    counter = 0
    for (x, y), value in np.ndenumerate(T):
        if not np.isnan(T[x, y]):
            T_vals.append(T[x, y])
            i += 1
            if Z[i-1] == T_vals[i-1]:
                counter += 1
    return np.sqrt(mean_squared_error(T_vals, Z)), counter/i

if alg == 'SGD':
    rmse_ = sgd_rmse(SGD(train_index, test_index, train_rating, test_rating), test.values)[0]
else:
    match alg:
        case 'NMF':
            Z = NMF_my(Z)
        case 'SVD1':
            Z = SVD(Z)
        case 'SVD2':
            Z = SVD_iterated(Z)

    Z = fill_with_original_values(orginal_data=training, edited_data=Z)
    Z = np.clip(Z, 1, 5)
    rmse_ = RMSE(round_and_clip(Z), test.values)



logger.info(str(rmse_))