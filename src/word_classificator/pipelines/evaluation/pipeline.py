from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=predict_test_data,
            inputs=["trained_classifier", "x_test"],
            outputs="predictions_x_test",
            name="predict_test_data_node"
        ),
        node(
            func=evaluate_predictions,
            inputs=["predictions_x_test", "y_test"],
            outputs="model_metrics_dict",
            name="evaluate_predictions_node"
        )
    ])
