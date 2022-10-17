from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from word_classificator.pipelines import text_extraction, data_preprocessing, load_spacy_model


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()

    # add parametrized pipelines
    pipelines["text_extraction_wikipedia_pipeline"] = text_extraction.create_pipeline(dataset="wikipedia")

    pipelines["__default__"] = sum(pipelines.values())
    pipelines["data_preprocessing"] = load_spacy_model.create_pipeline() + data_preprocessing.create_pipeline()
    return pipelines
