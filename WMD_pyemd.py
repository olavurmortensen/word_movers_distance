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
        d.append(document.count(t) / float(norm))

    return d

def WMD(document1, document2, embeddings, vocab):
    '''
    Compute WMD.

    Input:
    document1:      List of words.
    document2:      List of words.
    embeddings:     word2vec embeddings of words.
    vocab:          Set of words in all documents.

    Returns:        WMD between documents, float.
    '''

    # Compute nBOW representation of documents.
    d1 = np.array(nBOW(document1, vocab))
    d2 = np.array(nBOW(document2, vocab))

    distance_matrix = np.zeros((len(vocab), len(vocab)), dtype=np.float)
    for i, t1 in enumerate(vocab):
        for j, t2 in enumerate(vocab):
            distance_matrix[i][j] = distance(embeddings[t1], embeddings[t2])

    # Return WMD.
    return emd(d1, d2, distance_matrix)

if __name__ == '__main__':
    with open('w2v_model.pickle', 'rb') as f:
        model = pickle.load(f)

    # Sentence to compute distance between.
    sentence1 = 'one two five'.split()
    sentence2 = 'three four five'.split()

    # Remove out-of-vocabulary words.
    sentence1 = [token for token in sentence1 if token in model.vocab.keys()]
    sentence2 = [token for token in sentence2 if token in model.vocab.keys()]

    vocab = set()
    for token in sentence1 + sentence2:
        vocab.add(token)

    # Compute WMD.
    D = WMD(sentence1, sentence2, model, vocab)

    print D

