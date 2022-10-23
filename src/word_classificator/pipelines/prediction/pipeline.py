from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(dataset='test') -> Pipeline:
    """ Create prediction pipeline.

    :param dataset: if dataset is 'test' use x_test as input, else use 'words_to_predict'
    :return: pipeline
    """
    if dataset == "test":
        prediction_data = "x_test"
        output = "predictions_x_test"
    else:
        prediction_data = "words_to_predict"
        output = "predictions"

    return pipeline([
        node(
            func=make_predictions,
            inputs=prediction_data,
            outputs=output,
            name="make_predictions_node"
        )
    ])
