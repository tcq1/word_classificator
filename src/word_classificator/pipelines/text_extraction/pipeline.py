from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(dataset='pdf') -> Pipeline:
    """ Create the text extraction pipeline, which generates the basis for feature extraction.

    :param dataset: either 'pdf', 'wikipedia' or 'news'
    :return: pipeline
    """
    if dataset == 'pdf':
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
                outputs=["token_word_counts_pdf", "lemma_word_counts_pdf"],
                name="count_token_appearances_pdf_node"
            ),
        ])
    elif dataset == 'wikipedia':
        return pipeline([
            node(
                func=get_wikipedia_articles,
                inputs=["params:num_wikipedia_articles", "params:wikipedia_language"],
                outputs="wikipedia_dataset",
                name="get_wikipedia_articles_node"
            ),
            node(
                func=count_token_appearances,
                inputs=["wikipedia_dataset", "spacy_model"],
                outputs=["token_word_counts_wikipedia", "lemma_word_counts_wikipedia"],
                name="count_token_appearances_wikipedia_node"
            ),
        ])
    else:
        raise NotImplementedError(f"Pipeline parameter {dataset} not implemented!")
