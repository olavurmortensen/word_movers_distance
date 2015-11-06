#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate some word embeddings using word2vec.
"""

from time import time

from nltk.corpus import brown
from nltk import word_tokenize

import pickle

from gensim.models import Word2Vec

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class SentenceGenerator(object):
    '''
    Sentence generator using the Brown corpus from NLTK.

    Input:  num_sents; integer; number of sentences.
    '''
    def __init__(self, num_sents):
        self.brown_sentences = brown.sents()
        self.num_sents = num_sents

    def __iter__(self):
        i = 0
        for sentence in self.brown_sentences:
            if i > self.num_sents:
                break
            i += 1
            tokens = word_tokenize(' '.join(sentence))
            words = [w.lower() for w in tokens if w.isalnum()]
            yield words

if __name__ == '__main__':

    num_sents = 100000
    sentences = SentenceGenerator(num_sents=num_sents)

    logging.info('Training model.')
    start = time()
    model = Word2Vec(sentences, workers=3)
    logging.info('------------------------------------------------------')
    logging.info('Done training model. Time elapsed: %f seconds.', time() - start)

    pickle.dump(model, open('w2v_model.pickle', 'wb'))
