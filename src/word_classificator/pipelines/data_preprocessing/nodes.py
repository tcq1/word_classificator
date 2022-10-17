from typing import Dict, List

import numpy as np
import pandas as pd
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

    # get word vectors
    words = labeled_dataset["Word"]

    feature_vectors = []

    for word in words:
        if type(word) is not str:
            continue
        feature_vectors.append(get_feature_vector_for_word(word, features_to_use, spacy_model, document_counts_pdf, document_counts_wikipedia))

    return np.array(feature_vectors)


def get_feature_vector_for_word(word: str, features_to_use: List, spacy_model,
                                document_counts_pdf: List[Dict], document_counts_wikipedia: List[Dict]) -> np.array:
    """ Get the feature vector for a single word.

    :param word: word to extract features from
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

    return np.array(feature_values)
