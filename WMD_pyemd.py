#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute the WMD between two documents.
"""

from time import time
import pickle
import sys
import pdb

import numpy as np

from pyemd import emd

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def distance(v1, v2):
    '''
    Compute the Euclidean distance between two vectors v1 and v2.

    Input:
    v1, v2:     Numpy arrays (flat).

    Returns:    Distance (float).
    '''
    dist = 0
    for i in xrange(len(v1)):
        dist += (v1[i] - v2[i])**2
    
    dist = np.sqrt(dist)

    return dist

def nBOW(document, vocab):
    '''
    Compute nBOW representation of a document

    Input:
    doc:        Document, list of words (strings).
    vocab:      Set of words in all documents.

    Returns:    List of floats, nBow representation.
    '''

    norm = len(document)
    d = []
    for i, t in enumerate(vocab):
        d.append(document.count(t) / float(norm))  # TODO: when norm is zero?

    return d

def WMD(document1, document2, model):
    '''
    Compute WMD.

    Input:
    document1:      List of words.
    document2:      List of words.
    model:          Word2vec model, providing the word embeddings.
    vocab:          Set of words in all documents.

    Returns:        WMD between documents, float.
    '''

    # Remove out-of-vocabulary words.
    document1 = [token for token in document1 if token in model.vocab.keys()]
    document2 = [token for token in document2 if token in model.vocab.keys()]

    if len(document1) == 0 or len(document2) == 0:
        logging.info('At least one of the documents had no words that were in the vocabulary. Aborting.')
        return float('nan')

    vocab = set()
    for token in document1 + document2:
        vocab.add(token)

    # Compute nBOW representation of documents.
    d1 = np.array(nBOW(document1, vocab))
    d2 = np.array(nBOW(document2, vocab))

    distance_matrix = np.zeros((len(vocab), len(vocab)), dtype=np.float)
    for i, t1 in enumerate(vocab):
        for j, t2 in enumerate(vocab):
            distance_matrix[i][j] = distance(model[t1], model[t2])

    # Return WMD.
    return emd(d1, d2, distance_matrix)

if __name__ == '__main__':
    with open('w2v_model.pickle', 'rb') as f:
        model = pickle.load(f)

    # Sentence to compute distance between.
    sentence1 = 'one two five'.split()
    sentence2 = 'three four five'.split()
    
    # Compute WMD.
    D = WMD(sentence1, sentence2, model)

    print D

