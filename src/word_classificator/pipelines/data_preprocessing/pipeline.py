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
            func=extract_features,
            inputs=["positive_samples", "params:features", "spacy_model",
                    "lemma_word_counts_pdf", "lemma_word_counts_wikipedia"],
            outputs="positive_samples_features",
            name="extract_features_positive_samples_node"
        ),
        node(
            func=extract_features,
            inputs=["negative_samples", "params:features", "spacy_model",
                    "lemma_word_counts_pdf", "lemma_word_counts_wikipedia"],
            outputs="negative_samples_features",
            name="extract_features_negative_samples_node"
        ),
        node(
            func=merge_feature_vectors,
            inputs=["positive_samples_features", "negative_samples_features"],
            outputs="merged_feature_vectors",
            name="merge_feature_vectors_node"
        ),
        node(
            func=split_data,
            inputs=["merged_feature_vectors", "params:train_split_ratio"],
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
