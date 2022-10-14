import spacy


def get_spacy_model():
    """ Load the default spacy model for NLP tasks.

    :return: spacy model
    """
    return spacy.load('de_core_news_lg')
