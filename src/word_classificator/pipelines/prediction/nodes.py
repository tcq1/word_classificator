import numpy as np


def make_predictions(classifier, data: np.ndarray) -> np.ndarray:
    """ Use a classifier to perform predictions on the test dataset.

    :param classifier: classifier
    :param data: data to predict
    :return: predictions
    """
    return classifier.predict(data)
