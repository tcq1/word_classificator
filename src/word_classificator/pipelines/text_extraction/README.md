# Pipeline text_extraction

## Overview

This pipeline contains functionality to generate datasets by extracting texts from
either PDF files or random wikipedia articles. Also, so-called Document Count Datasets
are created based on those datasets, which count the appearance of words per document,
which can be further used as a feature for the classifier.

In order to process PDF files it is necessary to place the files in ``data/01_raw``.

## Pipeline inputs

* ``spacy_model``

## Pipeline outputs

* ``token_word_counts_pdf``
* ``lemma_word_counts_pdf``
* ``token_word_counts_wikipedia``
* ``lemma_word_counts_wikipedia``
