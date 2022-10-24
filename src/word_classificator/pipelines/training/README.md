# Pipeline training

## Overview

This pipeline initializes a model and trains it. Different models can be configured
in ``conf/base/parameters/training.yml`` via the ``model_parameters`` entry. In order
to change the desired model change the input of the ``initialize_classifier`` node in
``src/word_classificator/pipelines/training/pipeline.py``.

The training parameter file also contains parameters to configure the training process.

## Pipeline inputs

* ``x_train``
* ``y_train``

## Pipeline outputs

* ``trained_classifier``
