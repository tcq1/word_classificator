from pathlib import Path
from typing import List, Dict

import pandas as pd
from kedro.framework.session import KedroSession


def callback_retrieve_tags(text: str) -> List[str]:
    """ Get the tags of a text. The tags are words that are positively classified by the classifier.

    :param text: string
    :return: list of positively labeled words
    """
    with KedroSession.create("word_classificator", project_path=Path.cwd()) as session:
        # set words to predict
        session.load_context().catalog.save("words_to_predict_raw", text)

        # call prediction pipeline
        session.run(pipeline_name="predict_text")

        # load result
        predicted_words = session.load_context().catalog.load("predicted_words_labeled")

    # filter positive words
    tags = predicted_words[predicted_words["Label"] == 1]

    return list(tags["Word"])


def callback_get_data_positives() -> List[str]:
    """ Get positively labeled samples that are currently used for training.

    :return: list of positively labeled words
    """
    with KedroSession.create("word_classificator", project_path=Path.cwd()) as session:
        # load positive data from catalog
        positive_samples_df = session.load_context().catalog.load("positive_samples")

    return list(positive_samples_df["Word"])


def callback_get_data_negatives() -> List[str]:
    """ Get negatively labeled samples that are currently used for training.

    :return: list of negatively labeled words
    """
    with KedroSession.create("word_classificator", project_path=Path.cwd()) as session:
        # load positive data from catalog
        negative_samples_df = session.load_context().catalog.load("negative_samples")

    return list(negative_samples_df["Word"])


def callback_train(training_data_positive: List[str], training_data_negative: List[str]) -> Dict:
    with KedroSession.create("word_classificator", project_path=Path.cwd()) as session:
        # set positive and negative samples in catalog
        catalog = session.load_context().catalog

        # create dataframes
        positive_samples_tuples = [[positive_sample, 1] for positive_sample in training_data_positive]
        negative_samples_tuples = [[negative_sample, 0] for negative_sample in training_data_negative]

        positive_samples_df = pd.DataFrame(data=positive_samples_tuples, columns=["Word", "Label"])
        negative_samples_df = pd.DataFrame(data=negative_samples_tuples, columns=["Word", "Label"])

        # save to catalog
        catalog.save("positive_samples", positive_samples_df)
        catalog.save("negative_samples", negative_samples_df)

        # call default pipeline starting from merge datasets node, which takes positive and negative samples as input
        session.run(from_nodes=["merge_datasets_node", "initialize_classifier_node"])

        # load model metrics from evaluation
        model_metrics = catalog.load("model_metrics_dict")

    return model_metrics


def callback_available() -> bool:
    return True
