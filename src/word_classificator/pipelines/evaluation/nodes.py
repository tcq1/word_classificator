from typing import Dict

import numpy as np
import mlflow
from sklearn.metrics import classification_report


def predict_test_data(classifier, x_test: np.ndarray) -> np.ndarray:
    """ Use a classifier to perform predictions on the test dataset.

    :param classifier: classifier
    :param x_test: test data
    :return: predictions
    """
    return classifier.predict(x_test)


def evaluate_predictions(y_predicted: np.ndarray, y_expected: np.ndarray) -> Dict:
    """ Compare predicted labels to expected labels and calculate different evaluation metrics.

    :param y_predicted: predicted labels
    :param y_expected: expected labels
    :return: Dict containing different metrics
    """
    # calculate metrics
    metrics = classification_report(y_expected, y_predicted, output_dict=True)

    # log metrics
    mlflow.log_metric('accuracy', metrics['accuracy'])
    mlflow.log_metric('macro_avg_precision', metrics['macro avg']['precision'])
    mlflow.log_metric('macro_avg_recall', metrics['macro avg']['recall'])
    mlflow.log_metric('macro_avg_f1', metrics['macro avg']['f1-score'])

    return metrics
