from sklearn.model_selection import train_test_split
from pandas import read_csv


def train_and_test(data, part=0.9):
    """
    :param data: dataframe containing data you are working on
    :param part: which percent of data you want to have in training matrix
    :return: None
    """
    # count_by_movie = data.groupby('movieId').count()['userId']
    # filtered_movie_ids = count_by_movie[count_by_movie >= 5].index
    #
    # data = data[data['movieId'].isin(filtered_movie_ids)]

    train, test = train_test_split(data, train_size=part, random_state=17, stratify=data['userId'])

    train.to_csv('train_ratings.csv', index=False)
    test.to_csv('test_ratings.csv', index=False)


if __name__ == "__main__":
    data = read_csv('ratings.csv')
    train_and_test(data)
