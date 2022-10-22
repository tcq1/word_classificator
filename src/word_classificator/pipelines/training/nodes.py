from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import mlflow


def initialize_classifier(model_params: Dict):
    """ Initialize a random forest classifier with specified hyperparameters.

    :param model_params: kwargs containing model type and hyperparameters, if not specified then default values are used
    :return: classifier
    """
    # load classifier type
    model_type = model_params['type'].lower()
    del model_params['type']

    if model_type == 'rfc':
        mlflow.log_param("model", "rfc")
        return RandomForestClassifier(model_params)
    elif model_type == 'knn':
        mlflow.log_param("model", "knn")
        return KNeighborsClassifier(model_params)
    elif model_type == 'svm':
        mlflow.log_param("model", "svm")
        return SVC(model_params)
    else:
        raise NotImplementedError(f"Classifier {model_type} has not been implemented!")


def train_classifier(clf, x: np.ndarray, y: np.ndarray, k: int, scoring_method: str):
    """ Train a classifier using K-Fold cross validation.

    :param clf: classifier
    :param x: features for model input
    :param y: labels matching features
    :param k: k parameter for k-fold cross validation
    :param scoring_method: scoring method to evaluate model during training
    :return: trained classifier
    """
    # train model
    score = cross_val_score(clf, x, y, cv=k, scoring=scoring_method)

    # log metrics
    mlflow.log_metric("training_score_mean", score.mean())
    mlflow.log_metric("training_score_deviation", score.std())

    return clf
