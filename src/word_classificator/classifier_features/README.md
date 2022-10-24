# Classifier Features Package

## Overview
This package contains all functions that can be used to extract features from words.
To add new features, add the respective feature extraction function to features.py, as well as an entry in
``conf/base/parameters/data_preprocessing > features``, and extend the function ``get_feature_vector_for_word``
in ``src/word_classificator/pipelines/data_preprocessing/nodes.py``