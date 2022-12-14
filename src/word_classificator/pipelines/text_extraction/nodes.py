import io
import logging
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import pdfminer.pdfdocument
import wikipedia
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage

log = logging.getLogger(__name__)


def cleanup_data(root_dir: str, target_dir: str, extensions: List[str]):
    """ Copy all files in all subdirectories of root_dir, to target_dir that are in extension.

    :param root_dir: root directory
    :param target_dir: target directory
    :param extensions: extensions to not remove
    :return: bool determining success
    """
    # if target dir doesn't exist create
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # check subtree of root
    for root, directories, filenames in os.walk(root_dir, topdown=False):
        # iterate over files
        for filename in filenames:
            # copy files with correct extension to target directory
            filepath = os.path.join(root, filename).replace("\\", "/")
            try:
                ext = filename.split(".")[1]
                if ext in extensions:
                    shutil.copyfile(filepath, f"{target_dir}/{filename}")
            except IndexError:
                log.info(f"Couldn't get extension of file {filepath}. Skipping...")
            except FileNotFoundError:
                log.info(f"Couldn't find file {filepath}. Skipping...")

    return True


def pdf_to_string(path):
    """ Converts a pdf file to a string. String contains

    :param path: Path to pdf file
    :return: String
    """
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    with open(path, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)

        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()

    return text


def replace_cid_codes(string):
    """ Takes a string and replaces relevant cid codes

    :param string: string with cid codes
    :return: string with relevant cid codes replaced
    """
    # letters
    string = string.replace('(cid:228)', '??')
    string = string.replace('(cid:246)', '??')
    string = string.replace('(cid:252)', '??')

    string = string.replace('(cid:214)', '??')
    string = string.replace('(cid:220)', '??')
    string = string.replace('(cid:223)', '??')

    string = string.replace('\n', ' ')
    string = string.replace('\r', '')

    return string


def convert_pdfs_to_text(pdf_root_dir: str, _) -> List[str]:
    """ Convert all pdf files to a list of texts.

    :param pdf_root_dir: root directory containing pdf files
    :param _: placeholder to ensure correct pipeline execution order
    :return: Dict with filenames as keys and data as value
    """
    dataset = []

    # iterate over files in pdf_root_dir
    for root, directories, filenames in os.walk(pdf_root_dir, topdown=False):
        # convert and preprocess text file
        for filename in filenames:
            filepath = os.path.join(root, filename)
            try:
                text = pdf_to_string(filepath)
                processed_text = replace_cid_codes(text)

                # add to dataset
                dataset.append(processed_text)
                log.info(f"Done with processing file {filename}!")
            except FileNotFoundError:
                log.info(f"Couldn't find file {filepath}. Skipping...")
            except pdfminer.pdfdocument.PDFTextExtractionNotAllowed:
                log.info(f"Couldn't extract text from {filepath}. Skipping...")

    return dataset


def add_element_to_dict(dictionary, element):
    """ Adds an element to a dictionary. If not in dictionary, adds a new key to dictionary.

    :param dictionary: dictionary
    :param element: string
    :return: updated dictionary
    """

    if len(element) == 1:
        return dictionary

    if element not in dictionary.keys():
        dictionary[element] = 1
    else:
        dictionary[element] += 1

    return dictionary


def get_tokens(spacy_model, text):
    """ Extracts tokens from a text and returns a dictionary with the tokens and the number of appearances and
    a dictionary with the lemmas of the tokens.

    :param spacy_model: spacy model
    :param text: Text to extract the tokens from
    :return: word_dict, lemma_dict
    """
    word_dict = {}
    lemma_dict = {}

    doc = spacy_model(text)
    for token in doc:
        if token.is_alpha:
            add_element_to_dict(word_dict, token.text)
            add_element_to_dict(lemma_dict, token.lemma_)

    # sort dictionaries descending by appearance of tokens
    word_dict = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1], reverse=True)}
    lemma_dict = {k: v for k, v in sorted(lemma_dict.items(), key=lambda item: item[1], reverse=True)}

    return word_dict, lemma_dict


def count_token_appearances(texts: List[str], spacy_model) -> Tuple[List, List]:
    """ Convert each text into a Dict containing the tokens and their number of appearances (same with lemmas).

    :param texts: list of texts
    :param spacy_model: spacy model
    :return: dict with counts for tokens, dict with counts for lemmas of tokens
    """
    token_count = []
    lemma_count = []

    # iterate over texts
    for text in texts:
        tokens, lemmas = get_tokens(spacy_model, text)
        token_count.append(tokens)
        lemma_count.append(lemmas)

    return token_count, lemma_count


def get_wikipedia_articles(num_pages: int, language: str) -> List[str]:
    """ Get a list of the content of random wikipedia articles.

    :param num_pages: number of wikipedia pages to load
    :param language: language of articles
    :return: list of article content (texts)
    """
    wikipedia.set_lang(language)

    articles = []

    # find valid random articles
    counter = 0
    while counter < num_pages:
        try:
            articles.append(wikipedia.page(title=wikipedia.random()).content)
            counter += 1
            log.info(f"{counter}/{num_pages}: Downloading {articles[-1].title}")
        except wikipedia.DisambiguationError:
            # skip if DisambiguationError appears and decrement counter
            log.info('DisambiguationError! Skip...')
            continue
        except wikipedia.PageError:
            # skip if PageError appears and decrement counter
            log.info('PageError! Skip...')
            continue

    return articles
