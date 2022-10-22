from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=initialize_classifier,
            inputs="params:model_parameters.rfc",
            outputs="initialized_classifier",
            name="initialize_classifier_node"
        ),
        node(
            func=train_classifier,
            inputs=["initialized_classifier", "x_train", "y_train", "params:kfcv_k", "params:kfcv_scoring"],
            outputs="trained_classifier",
            name="train_classifier_node"
        )
    ])
