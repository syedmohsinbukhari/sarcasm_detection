# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:00:00 2018

@author: elcid
"""

""" Setup logging and environment """
# simulate that sarcasmdetection is installed as a python package
import context

"""--------------------------------------------------"""
import torch
import torchtext

import numpy as np
import logging
import json
import time
import sys
import pickle

import sarcasmdetection as sd

"""--------------------------------------------------"""
sd.utils.setup_logging('logs/predict.log')
logging.info("Running script predict.py")

"""--------------------------------------------------"""
utterances = []
labels = []

with open('data/main/final_data.json') as f:
    for line in f:
        cmnt_dict = json.loads(line)
        cmnt_id = list(cmnt_dict.keys())[0]
        cmnt_txt = cmnt_dict[cmnt_id]['text']
        cmnt_label = int(cmnt_dict[cmnt_id]['label'])
        utterances.append(cmnt_txt)
        labels.append(cmnt_label)

utterances = [[word for word in utterance.split()] for utterance in utterances]

test_indices = []
fname = sd.utils.get_biggest_fname('data/output/test_indices')
logging.info("Loading test set from file: " + fname)
with open(fname) as f:
    test_indices = json.load(f)

test_utterances = [utterances[i] for i in test_indices]
test_labels = [labels[i] for i in test_indices]

"""--------------------------------------------------"""
inputs = torchtext.data.Field(lower=True, include_lengths= True,
                              batch_first=True,
                              tokenize=torchtext.data.get_tokenizer('spacy'))
inputs.build_vocab(utterances)

emb_dim = 100
inputs.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=emb_dim))

test_numerized_inputs, seq_len_test = inputs.process(test_utterances,
                                                        device=-1, train=False)

"""--------------------------------------------------"""
def infer_accuracy(model, labels, numerized_inputs, seq_len):
    with torch.no_grad():
        log_scores, hidden_final = model(numerized_inputs, seq_len)
        scores = np.exp(log_scores)
        pred_labels = np.argmax(scores.numpy(), axis=1)
        test_labels = np.array(labels)

        accuracy = sd.utils.compute_accuracy(pred_labels, test_labels)
        return accuracy, hidden_final

"""--------------------------------------------------"""
torch.device("cuda")

model = torch.load('data/models/GRUClassifier_new_backup_3.dat')

accuracy, hidden_out = infer_accuracy(model, test_labels,
                                      test_numerized_inputs, seq_len_test)

logging.info(accuracy)

logging.info("Finished script predict.py")
