import spacy


def get_spacy_model(model_to_load: str):
    """ Load the default spacy model for NLP tasks.

    :param model_to_load: spacy model to load
    :return: spacy model
    """
    return spacy.load(model_to_load)
