from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=cleanup_data,
            inputs=["params:raw_data_dir", "params:pdf_data_dir", "params:allowed_extensions"],
            outputs=None,
            name="cleanup_raw_data_node"
        ),
        node(
            func=convert_pdfs_to_text,
            inputs=["params:pdf_data_dir"],
            outputs="text_dataset",
            name="convert_pdfs_to_text_node"
        )
    ])
