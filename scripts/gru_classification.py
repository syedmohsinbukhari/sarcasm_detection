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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import logging
import json
import time
import sys

import sarcasmdetection as sd

"""--------------------------------------------------"""
sd.utils.setup_logging('logs/gru_classification.log')
logging.info("Running script gru_classification.py")

"""--------------------------------------------------"""
utterances = []
labels = []

with open('data/pol/final_data.json') as f:
    comments = json.load(f)
    for k in comments.keys():
        cmnt_txt = comments[k]["text"]
        utterances.append(cmnt_txt)

        cmnt_lbl = int(comments[k]["label"])
        labels.append(cmnt_lbl)

utterances = [[word for word in utterance.split()] for utterance in utterances]
all_utterances = utterances
all_labels = labels

utterances = all_utterances[0:6500]
labels = all_labels[0:6500]

test_utterances = all_utterances[-20:]
test_labels = all_labels[-20:]

"""--------------------------------------------------"""
inputs = torchtext.data.Field(lower=True, include_lengths= True,
                              batch_first=True,
                              tokenize=torchtext.data.get_tokenizer('spacy'))
inputs.build_vocab(utterances)

emb_dim = 100
inputs.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=emb_dim))
numerized_inputs, seq_len = inputs.process(utterances, device=-1, train=True)

"""--------------------------------------------------"""
batch_sz = 20
epochs = 10
word_emd_sz = 100
disp_size = 100
vocab_sz = len(inputs.vocab)
model = sd.nnmodels.GRUClassifier(100, word_emd_sz, vocab_sz,
                                  inputs.vocab.vectors,
                                  2, batch_sz)

loss_function = nn.NLLLoss()
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

cnt = 0
loss = float('inf')
start_time = time.time()
for epoch in range(epochs):
    flg=False
    for start in range(0, len(utterances), batch_sz):
        cnt += 1
        if cnt%disp_size == 0:
            eta = ((time.time()-start_time)/(cnt*batch_sz))*\
                                    (epochs*len(utterances)-(cnt*batch_sz))/60
            eta_m = np.floor(eta)
            eta_s = (eta - eta_m) * 60
            perc = ((cnt*batch_sz)/(epochs*len(utterances)))*100
            out_str = "Progress: {:0.2f}%, ETA: {:0.0f}m{:0.0f}s".format(
                                                            perc, eta_m, eta_s)
            logging.info(out_str)
            logging.info("loss: "+str(loss))
            if loss.detach().numpy() < 0.1:
                break

        model.zero_grad()

        model.hidden = model.init_hidden()

        if start+batch_sz > len(utterances):
            break
        sentence_in = numerized_inputs[start:start + batch_sz]
        targets = torch.tensor(labels[start:start + batch_sz], dtype=torch.long)
        len_in = seq_len[start:start + batch_sz]

        log_scores = model(sentence_in, len_in)
        loss = loss_function(log_scores, targets)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    for start in range(0, len(test_utterances), batch_sz):
        numerized_inputs, seq_len = inputs.process(
                                        test_utterances, device=-1, train=False)

        scores = np.exp(model(numerized_inputs[start:start + batch_sz],
                                               seq_len[start:start + batch_sz]))
        pred_lables = np.argmax(scores.numpy(), axis=1)
        logging.info(list(pred_lables))
        logging.info(labels[:batch_sz])
