#!/usr/bin/env python

import re
import nltk

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

import pandas as pd
import numpy as np

# from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import tensorflow as tf
from tensorflow import keras

class NLPUtility(object):
    """NLPUtility is a utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def text_to_word_sequence(text):
        return keras.preprocessing.text.text_to_word_sequence(text, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ')

    @staticmethod
    def tokenize_sentence(sentence):
        return nltk.word_tokenize(sentence)

    @staticmethod
    def stem_words_ignore(words, ignore_words):
        words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
        return words

    @staticmethod
    def stem_words(words):
        words = [stemmer.stem(w.lower()) for w in words]
        return words

    @staticmethod
    def text_to_wordlist( text, remove_stopwords=False, stem=False ):
        # 1. Remove HTML
        # beautiful_text = BeautifulSoup(text, "lxml").get_text()
        #
        # 2. Remove non-letters
        beautiful_text = re.sub("[^a-zA-Z]"," ", text)
        beautiful_text = beautiful_text.lower()
        words = nltk.word_tokenize(beautiful_text)

        # 3. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]

        # 4. Optionally Stem each word
        if stem:
            words = [stemmer.stem(word.lower()) for word in words]

        # 5. Return a list of words
        return(words)

    # Define a function to split a text into parsed sentences
    @staticmethod
    def text_to_sentences( text, tokenizer, remove_stopwords=False ):
        # Function to split a text into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(text.decode('utf8').strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call text_to_wordlist to get a list of words
                sentences.append( NLPUtility.text_to_wordlist( raw_sentence, \
                  remove_stopwords ))
        #
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences
