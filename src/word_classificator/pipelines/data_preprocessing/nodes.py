from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from word_classificator.classifier_features.features import *


def merge_datasets(*datasets: List[Dict]) -> List[Dict]:
    """ Merge multiple datasets to one. A dataset in this case is a list of dicts (see document_count_dataset).

    :param datasets: list of datasets
    :return: merged dataset
    """
    merged_dataset = []

    for dataset in datasets:
        merged_dataset.extend(dataset)

    return merged_dataset


def generate_positive_samples(dataset: List[Dict]) -> pd.DataFrame:
    """ Extract all words of a dataset and label them as positive.

    :param dataset: document count dataset
    :return: dataframe of words and their labels
    """
    return generate_labeled_samples(dataset, True)


def generate_negative_samples(dataset: List[Dict]) -> pd.DataFrame:
    """ Extract all words of a dataset and label them as negative.

    :param dataset: document count dataset
    :return: dataframe of words and their labels
    """
    return generate_labeled_samples(dataset, False)


def generate_labeled_samples(dataset: List[Dict], label: bool) -> pd.DataFrame:
    """ Extract all words of a dataset and label them all with the same label (True: 1, False: 0).

    :param dataset: document count dataset
    :param label: boolean
    :return: dataframe of words and their labels
    """
    # get words from dataset
    words = set()
    for document in dataset:
        for word in document.keys():
            words.add(word)

    # create dataframe
    df = pd.DataFrame(data=[(word, int(label)) for word in words], columns=["Word", "Label"])
    df.dropna(inplace=True)

    return df


def extract_features(labeled_dataset: pd.DataFrame, features: Dict, spacy_model,
                     document_counts_pdf: List[Dict], document_counts_wikipedia: List[Dict]) -> np.array:
    """ Generate the feature vector for each word of a labeled dataset and collect the feature vectors in a dataframe.

    :param labeled_dataset: dataframe containing a 'word' column and a 'label' column
    :param features: configurable parameter determining the features to use
    :param spacy_model: nlp model
    :param document_counts_pdf: document word count dataset for pdf files
    :param document_counts_wikipedia: document word count dataset for wikipedia articles
    :return: dataframe with the feature vectors of each word
    """
    # extract features to use
    features_to_use = []
    for feature_name in features.keys():
        if features[feature_name]:
            features_to_use.append(feature_name)

    # get list of feature vectors
    feature_vectors = [get_feature_vector_for_word(word, class_label, features_to_use, spacy_model, document_counts_pdf, document_counts_wikipedia)
                       for word, class_label in zip(str(labeled_dataset['Word']), labeled_dataset['Label'])]

    return np.array(feature_vectors)


def get_feature_vector_for_word(word: str, class_label: int, features_to_use: List, spacy_model, document_counts_pdf: List[Dict],
                                document_counts_wikipedia: List[Dict]) -> np.array:
    """ Get the feature vector for a single word.

    :param word: word to extract features from
    :param class_label: class label of feature vector
    :param features_to_use: list of feature functions
    :param spacy_model: nlp model used for nlp feature extraction
    :param document_counts_pdf: document word count dataset for pdf files
    :param document_counts_wikipedia: document word count dataset for wikipedia articles
    :return: array of features
    """
    feature_values = []

    for feature in features_to_use:
        if feature == "get_number_syllables":
            feature_values.append(get_number_syllables(word))
        elif feature == "get_word_length":
            feature_values.append(get_word_length(word))
        elif feature == "has_capital_letter":
            feature_values.append(has_capital_letter(word))
        elif feature == "appearance_per_doc_length":
            appearance_ratio_pdfs, appearances_pdfs = appearance_per_doc_length(word, document_counts_pdf)
            appearance_ratio_wikipedia, appearance_wikipedia = appearance_per_doc_length(word, document_counts_wikipedia)
            feature_values.extend([appearance_ratio_pdfs, appearances_pdfs, appearance_ratio_wikipedia, appearance_wikipedia])
        elif feature == "tf_idf":
            feature_values.append(tf_idf(word, document_counts_pdf))
            feature_values.append(tf_idf(word, document_counts_wikipedia))
        elif feature == "normed_word_vector":
            feature_values.append(normed_word_vector(word, spacy_model))
        elif feature == "get_suffix":
            feature_values.append(get_suffix(word, spacy_model))
        elif feature == "get_prefix":
            feature_values.append(get_prefix(word, spacy_model))
        elif feature == "is_stop_word":
            feature_values.append(is_stop_word(word, spacy_model))

    # add class label as last element
    feature_values.append(class_label)

    return np.array(feature_values)


def merge_feature_vectors(*vectors: np.ndarray):
    """ Merge multiple sets of feature vectors together

    :param vectors: feature vectors as numpy arrays
    :return: all feature vectors merged into one array
    """
    return np.concatenate(vectors)


def split_data(dataset: np.ndarray, train_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Extract features and labels of a dataset and split into train and test datasets.

    :param dataset: feature vectors with labels in last column
    :param train_size: size of test dataset
    :return: x_train, x_test, y_train, y_test
    """
    # extract features and labels from dataset
    features = dataset[:, 0:-1]
    labels = dataset[:, -1]

    # split
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=train_size, stratify=labels)

    return x_train, x_test, y_train, y_test


def normalize_data(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Normalize the feature vectors using Min-max normalization. Also returns the normalization vectors.

    :param x_train: feature vectors of training dataset
    :param x_test: feature vectors of test dataset
    :return: normalized x_train, normalized x_test, min_norm, max_norm
    """
    min_x = np.min(x_train, axis=0)
    max_x = np.max(x_train, axis=0)

    x_train = (x_train - min_x) / (max_x - min_x)
    x_test = (x_test - min_x) / (max_x - min_x)

    return x_train, x_test, min_x, max_x
