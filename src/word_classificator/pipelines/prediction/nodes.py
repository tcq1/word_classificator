import numpy as np
import pandas as pd


def make_predictions(classifier, data: np.ndarray) -> np.ndarray:
    """ Use a classifier to perform predictions on the test dataset.

    :param classifier: classifier
    :param data: data to predict
    :return: predictions
    """
    return classifier.predict(data)


def assign_labels_to_words(words: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """ Assign predicted labels to the original words.

    :param words: dataframe containing 'Word' column
    :param labels: labels array
    :return: dataframe with additional 'Label' column
    """
    words['Label'] = labels

    return words
