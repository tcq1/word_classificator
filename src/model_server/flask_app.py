import json
import os

from flask import Flask, request
from kedro.framework.startup import bootstrap_project

from model_server.callbacks import *


app = Flask(__name__)


@app.route("/")
def entry():
    """ Entry point of webapp. Can be used as healthcheck

    :return: Simple HTML text
    """
    html_string = """
    <p>Flask app is running! Available callbacks:</p>
    <ul>
        <li>/retrieve_training_data</li>
        <li>/tag?text=[TEXT]</li>
        <li>/start_training</li>
    </ul>
    """
    return html_string


@app.route("/retrieve_training_data", methods=['GET'])
def retrieve_training_data():
    """ Load the models current training data.

    :return: JSON with positive and negative samples
    """
    # bootstrap kedro project
    os.chdir("/home/kedro")
    bootstrap_project(Path.cwd())

    # get samples
    positive_samples = callback_get_data_positives()
    negative_samples = callback_get_data_negatives()

    # remove any non string elements
    positive_samples = [sample for sample in positive_samples if type(sample) == str]
    negative_samples = [sample for sample in negative_samples if type(sample) == str]

    return {
        "positives": ','.join(positive_samples),
        "negatives": ','.join(negative_samples)
    }


@app.route("/tag", methods=['GET', 'POST'])
def retrieve_tags():
    """ Use the model to retrieve the tags (positively labeled tokens) from a text.

    :return: list of positively labeled words
    """
    # get text from request
    text = request.args.get("text")

    # bootstrap kedro project
    os.chdir("/home/kedro")
    bootstrap_project(Path.cwd())

    tags = callback_retrieve_tags(text)

    return {
        "result": "success",
        "tags": ','.join(tags)
    }


@app.route("/start_training", methods=['GET', 'POST'])
def train_model():
    """ Take post request data as training data and train model on it.

    :return: JSON data containing model metrics
    """
    # get training data
    training_data = json.loads(request.get_json())

    # extract positive and negative samples
    positive_samples = training_data["positives"].split(",")
    negative_samples = training_data["negatives"].split(",")

    # bootstrap kedro project
    os.chdir("/home/kedro")
    bootstrap_project(Path.cwd())

    # call training
    metrics = callback_train(positive_samples, negative_samples)

    return {
        "res": "success",
        "metrics": metrics
    }


if __name__ == '__main__':
    app.run()
