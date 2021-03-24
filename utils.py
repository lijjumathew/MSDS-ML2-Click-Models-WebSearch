
from pyclick.utils.Utils import Utils
from pyclick.utils.YandexRelPredChallengeParser import YandexRelPredChallengeParser


def get_click_model(click_model_name):
    """
    Returns the click models object given its name.

    :param click_model_name: The name of a click model to instantiate.
    :returns: The click models object given its name.
    """
    return globals()[click_model_name]()


def get_train_test(search_sessions_path, search_sessions_num):
    """
    Reads the given number of search sessions from the given path
    and splits them into train and test sets with the ratio 3/1.
    Queries that are not present in the train set are removed from the test set.
    Returns the train/test sessions and queries.

    :param search_sessions_path: The path to the file with search sessions
        in the format of Yandex Relevance Prediction Challenge
        (http://imat-relpred.yandex.ru/en).
    :param search_sessions_num: The number of search sessions to consider.
    :returns: Returns train sessions, train queries (distinct), test sessions, test queries (distinct).
    """
    search_sessions = YandexRelPredChallengeParser().parse(search_sessions_path, search_sessions_num)

    train_test_split = int(len(search_sessions) * 0.75)
    train_sessions = search_sessions[:train_test_split]
    train_queries = Utils.get_unique_queries(train_sessions)

    test_sessions = Utils.filter_sessions(search_sessions[train_test_split:], train_queries)
    test_queries = Utils.get_unique_queries(test_sessions)

    return train_sessions, train_queries, test_sessions, test_queries
