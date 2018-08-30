
#!/usr/bin/env python

import pandas as pd
import numpy as np
import random

import os.path
from os import path
import pickle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from build_code.handlers.data_handler import DataHandler, DataSet

class ModelHandler(object):
    def __init__(self, CONFIG):
        self.name = "scikit"
        self.CONFIG = CONFIG
        self.data_handler = self.get_data_handler()
        class DataSets(object):
            pass
        self.datasets = DataSets()
        self.prepare_data()

    def get_data_handler(self):
        df = pd.read_csv(self.CONFIG["DATA_FILE_PATH"], header=0, delimiter=",")
        return DataHandler(df, "scikit")

    def prepare_data(self):
        X = self.data_handler.dataframe["utterances"].values
        Y = self.data_handler.dataframe["intent"].values
        print("Training Data Length: ", len(X))
        print("Training Data Target Length: ", len(Y))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10, random_state = 42)
        self.datasets.train = DataSet(X_train, Y_train)
        self.datasets.test = DataSet(X_test, Y_test)

    def create_model(self):
        global model
        from sklearn.linear_model import SGDClassifier
        model = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                                 ('clf-svm', SGDClassifier(loss='squared_loss', penalty='l2',alpha=1e-3, max_iter=5, random_state=42))])
        model = model.fit(self.datasets.train.utterances, self.datasets.train.intents)
        predicted = model.predict(self.datasets.test.utterances)
        # print("\n\nSVM Prediction: >>> ", predicted)
        print("SVM Performance: >>> ", np.mean(predicted == self.datasets.test.intents))
        # print(metrics.classification_report(Y_test, predicted, target_names=Y_test))
        saved_model = joblib.dump(model, self.CONFIG["MODEL_PATH"])
        print("<<<<<<<< ML MODEL CREATED AND SAVED >>>>>>>>>>>\n\n")
        return model

    def load_model(self):
        return joblib.load(self.CONFIG["MODEL_PATH"])

    def predict(self, texts):
        toPredict = pd.Series(texts)
        model = self.load_model()
        results = model.decision_function(toPredict)
        return results[0]
