
#!/usr/bin/env python

import pandas as pd
import numpy as np
import random
import json

import os.path
from os import path

import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input, concatenate, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
# from tf.keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
# from tf.keras.layers.core import Dropout
from keras import backend as K
from keras.models import load_model

from build_code.handlers.data_handler import DataHandler, DataSet

class ModelHandler(object):
    def __init__(self, CONFIG):
        self.name = "keras"
        self.CONFIG = CONFIG
        self.data_handler = self.get_data_handler()
        class DataSets(object):
            pass
        self.datasets = DataSets()

    def get_data_handler(self):
        # df = pd.read_csv('../../../data/raw_home_automation.csv', header=0, delimiter=",")
        df = pd.read_csv(self.CONFIG["DATA_FILE_PATH"], header=0, delimiter=",")
        return DataHandler(df, "keras")

    def prepare_data(self):
        X, Y = self.data_handler.get_training_data()
        print("Training Data Length: ", len(X))
        print("Training Data Target Length: ", len(Y))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10, random_state = 42)
        self.datasets.train = DataSet(X_train, Y_train)
        self.datasets.test = DataSet(X_test, Y_test)

    def create_model(self):
        self.prepare_data()
        K.clear_session()
        tf.reset_default_graph()
        init_g = tf.global_variables_initializer()
        init_l = tf.local_variables_initializer()
        with tf.Session() as sess:
          sess.run(init_g)
          sess.run(init_l)
          # Create the network definition based on Gated Recurrent Unit (Cho et al. 2014).
          embedding_vector_length = 32

          model = Sequential()
          model.add(Embedding(self.data_handler.max_features, embedding_vector_length, input_length=self.data_handler.maxlen))
          model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
          model.add(MaxPooling1D(pool_size=2))
          model.add(LSTM(100))
          model.add(Dense(len(self.datasets.train.intents[0]), activation=self.CONFIG["MODEL_CONFIG"]["activation"]))
          model.compile(loss=self.CONFIG["MODEL_CONFIG"]["loss"], optimizer=self.CONFIG["MODEL_CONFIG"]["optimizer"], metrics=self.CONFIG["MODEL_CONFIG"]["metrics"])
          print(model.summary())

          tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=self.CONFIG["LOG_DIR"], write_graph=True)

          monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=self.CONFIG["MODEL_CONFIG"]["patience"], verbose=0, mode='auto')
          # checkpointer = ModelCheckpoint(filepath=self.CONFIG["MODEL_WEIGHTS_PATH"], verbose=0, save_best_only=True) # Save best model
          model.fit(np.asarray(self.datasets.train.utterances), np.asarray(self.datasets.train.intents), epochs=self.CONFIG["MODEL_CONFIG"]["epochs"], batch_size=self.CONFIG["MODEL_CONFIG"]["batch_size"],  verbose=1, validation_split=0.02, callbacks=[tbCallBack, monitor])
          # model.load_weights(self.CONFIG["MODEL_WEIGHTS_PATH"]) # load weights from best model
          scores = model.evaluate(np.asarray(self.datasets.test.utterances), np.asarray(self.datasets.test.intents))
          print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
          model.save(self.CONFIG["MODEL_PATH"])
          print("<<<<<<<< ML MODEL CREATED AND SAVED LOCALLY AT: ", self.CONFIG["MODEL_PATH"])
          word_index_file = os.path.join(self.CONFIG["DATA_DIR"], 'word_index.json')
          with open(word_index_file, 'w') as outfile:
               json.dump(self.data_handler.get_tokenizer().word_index, outfile)
          print("word_index.json file uploaded successfully....")

    def load_keras_model(self):
        model = load_model(self.CONFIG["MODEL_PATH"])
        # model.load_weights(self.CONFIG["MODEL_WEIGHTS_PATH"]) # load weights from best model
        return model

    def predict(self, texts):
        ERROR_THRESHOLD = 0.15
        model = self.load_keras_model()
        toPredict = self.data_handler.convert_to_predict(texts)
        predictions = model.predict(np.asarray(toPredict))[0]
        # np.argmax(predictions[0])
        # filter out predictions below a threshold
        # predictions = [[i,r] for i,r in enumerate(predictions) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        # predictions.sort(key=lambda x: x[1], reverse=True)
        # print("predictions: >> ", predictions)
        # return_list = []
        # for r in predictions:
        #     return_list.append((self.data_handler.intents[r[0]], r[1]))
        # return tuple of intent and probability
        return predictions
