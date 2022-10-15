from typing import Any, Dict, List

from kedro.io import AbstractDataSet


class DocumentCountDataSet(AbstractDataSet[List]):
    """ Load/save so-called document count datasets, which are lists of documents, for which the number of tokens/lemmas/words are counted.

    Each document is a dictionary with the keys being the tokens and the values their number of appearances in the respective document.
    The document count dataset is a list of these documents.
    """

    def __init__(self, filepath: str):
        """ Creates a new instance of DocumentDataSet to load/store it at given filepath.

        :param filepath: location of dataset
        """
        self._filepath = filepath

    def _save(self, data: List[Dict]):
        """ Save the DocumentCountDataSet.

        :param data: DocumentCountDataSet
        """
        export_docs(data, self._filepath)

    def _load(self) -> List[Dict]:
        """ Load the data from the DocumentCountDataSet.

        :return: List of dictionaries
        """
        return import_docs(self._filepath)

    def _describe(self) -> Dict[str, Any]:
        """ Returns a dict describing this classes attributes.

        :return: attribute description dictionary
        """
        return {"filepath": self._filepath}


def export_docs(documents, output_path):
    """ Exports a list of dictionaries to a single csv file

    :param documents: list of dictionaries
    :param output_path: file path to csv file
    """

    for doc in documents:
        export_dict(doc, output_path)


def export_dict(dictionary, output_path):
    """ Exports an ordered dictionary to a csv file

    :param dictionary: dictionary with words as keys and the number of appearances as values
    :param output_path: file path to csv file
    """

    with open(output_path, 'a', encoding='utf-8') as f:
        for key in dictionary.keys():
            try:
                f.write('{}:{},'.format(key, dictionary[key]))
            except UnicodeEncodeError:
                print("Couldn't encode {}. Skip".format(key))
        f.write('\n')

    f.close()


def import_docs(file_path):
    """ Imports a csv file and returns a list of dictionaries

    :param file_path: Path of csv file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().splitlines()

    f.close()

    docs = []
    for line in content:
        docs.append(import_dict(line))

    return docs


def import_dict(line):
    """ Converts line of a csv file to a dict
    :param line: line of csv file
    :return: dictionary
    """

    dictionary = {}

    elements = line.split(',')
    for element in elements:
        try:
            key, value = element.split(':')
            dictionary[key] = int(value)
        except ValueError:
            pass

    return dictionary
