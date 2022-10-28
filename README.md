# word_classificator

## Overview

This project creates a machine learning model which is able to classify words regarding whether a word belongs
to a certain context or not. Therefore, the texts of PDF files form the base of the context, e.g. using a 
PDF collection of material testing will train a model that can determine if a word belongs to the domain of
material testing or not. Furthermore, the model can perform predictions for whole texts and find words belonging
to the trained domain.

## Kedro 
This project is based on [Kedro](https://kedro.readthedocs.io/en/stable/). For further information, feel free to visit Kedro's well documented documentation.

## spaCy
For different NLP tasks, such as tokenization, feature extraction, etc. [spaCy](https://spacy.io/) is used. It is necessary
to download a spaCy model in order to execute this projects pipelines. For further details visit the [spaCy models page](https://spacy.io/usage/models).

## MlFlow
This project uses [mlflow](https://mlflow.org/) for model tracking. mlflow comes with the tool mlflow ui, which makes it possible
to track the model's performance in a simple webinterface. This project uses the Kedro plugin ``kedro-mlflow``, which makes configuration easy via
the ``conf/base/mlflow.yml`` configuration file. To open mlflow ui run ``mlflow ui`` in the projects root (this directory).

## How to install dependencies

This project mainly uses [pyPoetry](https://python-poetry.org/) for the management of the Python environment. 
Dependencies are declared in the ``pyproject.toml``. To set up the environment run ``poetry install`` in the projects root (this directory).
If you want to add new dependencies, run ``poetry add <DEPENDENCY>``.

For further details visit the [documentation of pyPoetry](https://python-poetry.org/docs/).

## Project and Pipeline configuration
Kedro makes it easy to configure project and pipeline parameters. In ``conf/local`` credentials or information can be stored, which is not supposed to be added
to any kind of repository or docker image. In ``conf/base`` are configuration files for logging, mlflow and the Kedro catalog (for further details on the catalog visit 
[Kedro docs](https://kedro.readthedocs.io/en/stable/data/data_catalog.html)).
Kedro pipelines can be configured in the configuration files in ``conf/base/parameters``. For more details regarding pipeline parameters have a look at the pipelines README.md file,
which can be found in ``src/word_classificator/pipelines/<PIPELINE>``.

## How to run Kedro pipelines

In order to run Kedro pipelines simply run ``kedro pipeline run --pipeline=<PIPELINE NAME>`` in the projects root (this directory).
If the --pipeline flag is missing the default pipeline is run.

This Kedro project mainly supports two pipelines:

* ``Default``: Run this pipeline by executing ``kedro pipeline run``
  * This pipeline requires a collection of PDF files in ``data/01_raw``. The files can be in any kind of subdirectory as well.
  * The default pipeline extracts texts from PDF files and Wikipedia articles to create a base dataset for model training.
  * The dataset is preprocessed and split into train and test datasets
  * A model is initialized and trained on the training dataset
  * After training the model makes predictions for the test dataset, and it's performance is evaluated
  * The model can be further used in the ``Predict Text`` pipeline
  * The model metrics can be seen using MlFlow or under ``08_reporting/model_metrics/model_metrics.json`` 
* ``Predict Text``: Run this pipeline by executing ``kedro pipeline run --pipeline predict_text``
  * This pipeline requires a trained model as well as a text file containing text in ``data/04_feature/words_to_predict/words_to_predict.txt``
  * This pipeline extracts tokens from the specified text file, preprocesses them and makes predictions for each token.
  * The predictions can be found in ``data/07_model_output/predicted_words.csv``, where a label of 1 indicates that the word belongs to the trained domain.

## Flask Application
The Kedro pipelines can also be triggered using a [Flask](https://flask.palletsprojects.com/) application.
In order to run the flask application simply run ``flask --app flask_app run`` in the ``src/model_server`` directory.

## Docker Container
In order to deploy this project in a Docker container simply run ``kedro docker build`` to build an image.
The configured Dockerfile will run the flask application upon running the container to make the kedro pipelines accessible.

> ATTENTION: All files and directories in the *data* directory are copied into the container. This can be used to serve a pretrained model.
> The accessible pipelines can only train a model, load training data or make predictions. In order to have a functioning
> project it is necessary to have executed following steps *BEFORE* building the image:
> * Collect PDF files in ``data/01_raw``
> * Run following pipelines:
>   * load_spacy_model
>   * text_extraction
> * (Optional, can also be done once deployed) Train a model
> * Configure all parameters in ``conf/base/parameters``
