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


def nBOW(document, vocab):
    '''
    Compute nBOW representation of a document

    Input:
    doc:        Document, list of words (strings).
    vocab:      Set of words in all documents.

    Returns:    List of floats, nBow representation.
    '''

    doc_len = len(document)
    d = []
    for i, t in enumerate(vocab):
        d.append(document.count(t) / float(doc_len))

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
    len_pre_oov1 = len(document1)
    len_pre_oov2 = len(document2)
    document1 = [token for token in document1 if token in model.vocab.keys()]
    document2 = [token for token in document2 if token in model.vocab.keys()]
    logging.info('Removed %d and %d OOV words from document 1 and 2 (respectively).', len_pre_oov1 - len(document1), len_pre_oov2 - len(document2))

    if len(document1) == 0 or len(document2) == 0:
        logging.info('At least one of the documents had no words that were in the vocabulary. Aborting (returning NaN).')
        return float('nan')

    vocab = set(document1 + document2)

    # Compute nBOW representation of documents.
    d1 = np.array(nBOW(document1, vocab))
    d2 = np.array(nBOW(document2, vocab))

    vocab_len = len(vocab)
    distance_matrix = np.zeros((vocab_len, vocab_len), dtype=np.float)
    for i, t1 in enumerate(vocab):
        for j, t2 in enumerate(vocab):
            if not t1 in document1 or not t2 in document2:
                # Only compute the distances that we need.
                continue
            # Compute Euclidean distance between word vectors.
            # TODO: this matrix is (and should be) symmetric, so we can save some computation here.
            # TODO: why not cosine distance?
            distance_matrix[i][j] = np.sqrt(np.sum((model[t1] - model[t2])**2))

    # Return WMD.
    return emd(d1, d2, distance_matrix)

if __name__ == '__main__':
    with open('w2v_model.pickle', 'rb') as f:
        model = pickle.load(f)

    # Sentence to compute distance between.
    sentence1 = 'one two three'.split()
    sentence2 = 'four five one'.split()
    
    # Compute WMD.
    D = WMD(sentence1, sentence2, model)

    print D

