#!/usr/bin/env python

#  Author: Gurvinder Singh
#  Date: 01/08/2018
#
# Natural Language Classification.
#
# python build_code/IntentClassification.py --data_dir data --result_dir results --config_file model_config.json --data_file data.csv
#
# *************************************** #

"""A very simple NLC classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
from os import environ
import tarfile
import json

import pandas as pd
import numpy as np
import random

import tensorflow as tf

# from handlers.scikit_model_handler import ModelHandler
# from handlers.keras_model_handler import ModelHandler

FLAGS = None

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def set_config():
    # print(FLAGS)
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

    DATA_FILE_PATH = os.path.join(DATA_DIR, FLAGS.data_file)
    MODEL_PATH = os.path.join(RESULT_DIR, "model", MODEL_CONFIG["model_name"])
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
                "LOG_DIR": LOG_DIR,
                "MODEL_CONFIG": MODEL_CONFIG
             }

def get_keras_model():
    print("\n\n <<<<<<<< GET KERAS MODEL HANDLER >>>>>>>>")
    from handlers.keras_model_handler import ModelHandler
    model_handler = ModelHandler(CONFIG)
    return model_handler

def get_scikit_model():
    print("\n\n <<<<<<<< GET SCIKIT MODEL HANDLER >>>>>>>>")
    from handlers.scikit_model_handler import ModelHandler
    model_handler = ModelHandler(CONFIG)
    return model_handler

def get_model_handler():
    if FLAGS.framework == "scikit":
        return get_scikit_model()
    elif FLAGS.framework == "keras":
        return get_keras_model()
    else:
        return None

def create_model():
    model_handler = get_model_handler()
    print(model_handler.name)
    if FLAGS.framework == "scikit":
        print("\n\n <<<<<<<< CREATE MODEL FROM SCIKIT LIBRARY >>>>>>>>")
        model_handler.create_model()
    elif FLAGS.framework == "keras":
        print("\n\n <<<<<<<< CREATE MODEL FROM KERAS LIBRARY >>>>>>>>")
        model_handler.create_model()
    else:
        return None

def main(_):
    set_config()
    create_model()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # environment variable when name starts with $
  parser.add_argument('--data_dir', type=str, default='$DATA_DIR', help='Directory with data')
  parser.add_argument('--result_dir', type=str, default='$RESULT_DIR', help='Directory with results')
  parser.add_argument('--data_file', type=str, default='data.csv', help='Data file name')
  parser.add_argument('--config_file', type=str, default='model_config.json', help='Model Configuration file name')
  parser.add_argument('--framework', type=str, default='keras', help='ML Framework to use')

  FLAGS, unparsed = parser.parse_known_args()
  print("Start model training")
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
