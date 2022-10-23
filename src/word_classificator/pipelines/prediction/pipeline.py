from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(train_new_model=True) -> Pipeline:
    """ Create prediction pipeline.

    :param train_new_model: if true then predictions are made for x_test
    :return: pipeline
    """
    if not train_new_model:
        return create_text_prediction_pipeline()

    return pipeline([
        node(
            func=make_predictions,
            inputs=["trained_classifier", "x_test"],
            outputs="predictions_x_test",
            name="make_predictions_node"
        )
    ])


def create_text_prediction_pipeline():
    """ Create a pipeline that makes predictions for the words of a text.

    :return: pipeline
    """
    return pipeline([
        node(
            func=make_predictions,
            inputs=["trained_classifier", "words_to_predict"],
            outputs="predictions",
            name="make_predictions_node"
        ),
        node(
            func=assign_labels_to_words,
            inputs=["words_to_predict_df", "predictions"],
            outputs="predicted_words_labeled",
            name="assign_labels_to_words_node"
        )
    ])
