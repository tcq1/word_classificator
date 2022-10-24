from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=get_spacy_model,
            inputs="params:spacy_model_name",
            outputs="spacy_model",
            name="load_spacy_model_node"
        )
    ])
