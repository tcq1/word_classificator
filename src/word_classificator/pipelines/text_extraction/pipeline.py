from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=cleanup_data,
            inputs=["params:raw_data_dir", "params:pdf_data_dir", "params:allowed_extensions"],
            outputs="data_cleanup_done",
            name="cleanup_raw_data_node"
        ),
        node(
            func=convert_pdfs_to_text,
            inputs=["params:pdf_data_dir", "data_cleanup_done"],
            outputs="text_dataset",
            name="convert_pdfs_to_text_node"
        ),
        node(
            func=count_token_appearances,
            inputs=["text_dataset", "spacy_model"],
            outputs=["token_word_counts", "lemma_word_counts"],
            name="count_token_appearances_node"
        )
    ])
