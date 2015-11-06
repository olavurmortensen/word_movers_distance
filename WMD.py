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

PATH_TO_EMD = '../wmd/python-emd-master'
sys.path.append(PATH_TO_EMD)
from emd import emd

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

def nBOW(document):
    '''
    Compute nBOW representation of a document

    Input:
    doc:        Document, list of words (strings).

    Returns:    List of floats, nBow representation.
    '''

    # Compute the normalized frequency of each token (store in dictionary).
    norm = len(document)
    d_dict = {}
    for token in document:
        d_dict[token] = document.count(token) / float(norm)

    # Procude list with normalized frequency for each token in document.
    d = []
    for i, token in enumerate(document):
        d.append(d_dict[token])

    return d

def WMD(document1, document2, embeddings):
    '''
    Compute WMD.

    Input:
    document1:      List of words.
    document2:      List of words.
    embeddings:     word2vec embeddings of words.

    Returns:        WMD between documents, float.
    '''

    # Compute nBOW representation of documents.
    d1 = nBOW(document1)
    d2 = nBOW(document2)
    
    # Get features.
    features1 = [tuple(embeddings[token]) for token in document1]
    features2 = [tuple(embeddings[token]) for token in document2]

    # Return WMD.
    return emd((features1, d1), (features2, d1), distance)

if __name__ == '__main__':
    with open('w2v_model.pickle', 'rb') as f:
        model = pickle.load(f)

    # Sentence to compute distance between.
    sentence1 = 'mayor london'.split()
    sentence2 = 'president america'.split()

    # Remove out-of-vocabulary words.
    sentence1 = [token for token in sentence1 if token in model.vocab.keys()]
    sentence2 = [token for token in sentence2 if token in model.vocab.keys()]

    # Compute WMD.
    D = WMD(sentence1, sentence2, model)

    print D



    

