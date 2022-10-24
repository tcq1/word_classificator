# Pipeline data_preprocessing

## Overview

This pipeline preprocesses raw data containing functionality to generate labeled datasets,
clean datasets from invalid samples, extract features from words, normalize data and split
datasets for model training and evaluation.

## Pipeline inputs

* ``lemma_word_counts_pdf``
* ``lemma_word_counts_wikipedia``
* ``spacy_model``

## Pipeline outputs

* ``positive_samples``
* ``negative_samples``
* ``merged_word_dataset``
* ``cleaned_word_dataset``
* ``feature_vectors``
* ``x_train``
* ``x_test``
* ``y_train``
* ``y_test``
* ``norm_min``
* ``norm_max``
* ``words_to_predict_df``
* ``words_to_predict_features``
* ``words_to_predict``
