from kedro.pipeline import Pipeline, node, pipeline
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generate_positive_samples,
            inputs="lemma_word_counts_pdf",
            outputs="positive_samples",
            name="generate_positive_labeled_samples_node"
        ),
        node(
            func=generate_negative_samples,
            inputs="lemma_word_counts_wikipedia",
            outputs="negative_samples",
            name="generate_negative_labeled_samples_node"
        ),
        node(
            func=merge_datasets,
            inputs=["positive_samples", "negative_samples"],
            outputs="merged_word_dataset",
            name="merge_datasets_node"
        ),
        node(
            func=clean_dataset,
            inputs=["merged_word_dataset", "spacy_model"],
            outputs="cleaned_word_dataset",
            name="clean_dataset_node"
        ),
        node(
            func=extract_features,
            inputs=["cleaned_word_dataset", "params:features", "spacy_model",
                    "lemma_word_counts_pdf", "lemma_word_counts_wikipedia"],
            outputs="feature_vectors",
            name="extract_features_node"
        ),
        node(
            func=split_data,
            inputs=["feature_vectors", "params:train_split_ratio"],
            outputs=["x_train_not_normalized", "x_test_not_normalized", "y_train", "y_test"],
            name="split_data_node"
        ),
        node(
            func=normalize_data,
            inputs=["x_train_not_normalized", "x_test_not_normalized"],
            outputs=["x_train", "x_test", "norm_min", "norm_max"],
            name="normalize_data_node"
        )
    ])
