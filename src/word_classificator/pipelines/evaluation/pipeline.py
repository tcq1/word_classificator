from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=evaluate_predictions,
            inputs=["predictions_x_test", "y_test"],
            outputs="model_metrics_dict",
            name="evaluate_predictions_node"
        )
    ])
