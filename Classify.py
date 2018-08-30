
## python Classify.py --data_dir data --result_dir results --config_file model_config.json --data_file data.csv --from_cloud False

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
from os import environ
import tarfile

import pandas as pd
import numpy as np
import random
import re

import tensorflow as tf
import numpy as np
import json

import urllib3, requests, json, base64, time, os, wget
from watson_machine_learning_client import WatsonMachineLearningAPIClient

# from build_code.handlers.scikit_model_handler import ModelHandler
# from build_code.handlers.keras_model_handler import ModelHandler
from build_code.handlers.data_handler import DataHandler

FLAGS = None

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_config():
    if (FLAGS.data_dir[0] == '$'):
      DATA_DIR = os.environ[FLAGS.data_dir[1:]]
    else:
      DATA_DIR = FLAGS.data_dir
    if (FLAGS.result_dir[0] == '$'):
      RESULT_DIR = os.environ[FLAGS.result_dir[1:]]
    else:
      RESULT_DIR = FLAGS.result_dir

    with open(os.path.join(DATA_DIR, FLAGS.config_file), 'r') as f:
        MODEL_CONFIG = json.load(f)

    global SECRET_CONFIG
    with open('config.json', 'r') as f:
        SECRET_CONFIG = json.load(f)

    DATA_FILE_PATH = os.path.join(DATA_DIR, FLAGS.data_file)
    MODEL_PATH = os.path.join(RESULT_DIR, "model", MODEL_CONFIG["model_name"])
    MODEL_WEIGHTS_PATH = os.path.join(RESULT_DIR, "model", MODEL_CONFIG["model_weights"])
    if environ.get('JOB_STATE_DIR') is not None:
        LOG_DIR = os.path.join(os.environ["JOB_STATE_DIR"], MODEL_CONFIG["log_dir"])
    else:
        LOG_DIR = os.path.join(RESULT_DIR, MODEL_CONFIG["log_dir"])
    ensure_dir(DATA_FILE_PATH)
    ensure_dir(MODEL_PATH)
    global CONFIG
    CONFIG = {
                "DATA_DIR": DATA_DIR,
                "RESULT_DIR": RESULT_DIR,
                "DATA_FILE_PATH": DATA_FILE_PATH,
                "MODEL_PATH": MODEL_PATH,
                "MODEL_WEIGHTS_PATH": MODEL_WEIGHTS_PATH,
                "LOG_DIR": LOG_DIR,
                "MODEL_CONFIG": MODEL_CONFIG
             }

def get_keras_model():
    print("\n\n <<<<<<<< GET KERAS MODEL HANDLER >>>>>>>>")
    from build_code.handlers.keras_model_handler import ModelHandler
    model_handler = ModelHandler(CONFIG)
    return model_handler

def get_scikit_model():
    print("\n\n <<<<<<<< GET SCIKIT MODEL HANDLER >>>>>>>>")
    from build_code.handlers.scikit_model_handler import ModelHandler
    model_handler = ModelHandler(CONFIG)
    return model_handler

def get_model_handler():
    if FLAGS.framework == "scikit":
        return get_scikit_model()
    elif FLAGS.framework == "keras":
        return get_keras_model()
    else:
        return None

def get_scoring_url():
    # deployment_details = client.deployments.get_details(SECRET_CONFIG["deployment_id"]);
    # scoring_url = client.deployments.get_scoring_url(deployment_details)
    # print("scoring_url: >> ", scoring_url)
    scoring_url = 'https://ibm-watson-ml.mybluemix.net/v3/wml_instances/e7e44faf-ff8d-4183-9f37-434e2dcd6852/deployments/ab5304e7-cc30-44c0-b808-0d74044da792/online'
    return scoring_url

def convert_to_predict(text):
    preprocessed_records = []
    maxlen = 50

    cleanString = re.sub(r"[!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~]", "", text)
    splitted_text = cleanString.split()[:maxlen]
    hashed_tokens = []
    for token in splitted_text:
        # index = self.get_tokenizer().word_index.get(token, 0)
        index = scoring_params["word_index"].get(token, 0)
        if index < 501 and index > 0:
            hashed_tokens.append(index)

    hashed_tokens_size = len(hashed_tokens)
    padded_tokens = [0]*(maxlen - hashed_tokens_size) + hashed_tokens
    preprocessed_records.append(padded_tokens)
    return preprocessed_records

def get_results(sentence):
    ERROR_THRESHOLD = 0.15
    if FLAGS.from_cloud:
        toPredict = convert_to_predict(sentence)
        # if (to_predict_arr.ndim == 1):
        #     to_predict_arr = np.array([to_predict_arr])
        to_predict_arr = np.asarray(toPredict)
        scoring_data = {'values': to_predict_arr.tolist()}
        resp = client.deployments.score(scoring_params["scoring_endpoint"], scoring_data)
        result = resp["values"][0][0]
        result = [[i,r] for i,r in enumerate(result) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        result.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in result:
            return_list.append((scoring_params["intents"][r[0]], r[1]))
        return return_list
    else:
        result = model_handler.predict([sentence])
        result = [[i,r] for i,r in enumerate(result) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        result.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in result:
            return_list.append((model_handler.data_handler.get_intents()[r[0]], r[1]))
        return return_list

def init_content():
    set_config()
    print(FLAGS.from_cloud)
    global model_handler
    try:
      model_handler
    except NameError:
      model_handler = get_model_handler()
    if FLAGS.from_cloud:
        wml_credentials=SECRET_CONFIG["wml_credentials"]
        global client
        client = WatsonMachineLearningAPIClient(wml_credentials)
        scoring_endpoint = get_scoring_url()
        with open('data/word_index.json') as f:
            word_index = json.load(f)
        global scoring_params
        scoring_params = {
            "scoring_endpoint": scoring_endpoint,
            "intents": model_handler.data_handler.get_intents(),
            "word_index": word_index
        }


def classify(_):
    init_content()
    print("Model is ready! You now can enter requests.")
    for query in sys.stdin:
        if query.strip() == "close":
            sys.exit(0)
        print(get_results(query.strip()))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # environment variable when name starts with $
  parser.add_argument('--data_dir', type=str, default='$DATA_DIR', help='Directory with data')
  parser.add_argument('--result_dir', type=str, default='$RESULT_DIR', help='Directory with results')
  parser.add_argument('--framework', type=str, default='keras', help='ML Framework to use')
  parser.add_argument('--data_file', type=str, default='data.csv', help='File name for Intents and Classes')
  parser.add_argument('--config_file', type=str, default='model_config.json', help='Model Configuration file name')
  parser.add_argument('--from_cloud', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True, help='Predict value from model deployed on cloud')

  FLAGS, unparsed = parser.parse_known_args()
  print("Start model training")
  tf.app.run(main=classify, argv=[sys.argv[0]] + unparsed)
