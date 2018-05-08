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

"""--------------------------------------------------"""
inputs = torchtext.data.Field(lower=True, include_lengths= True,
                              batch_first=True,
                              tokenize=torchtext.data.get_tokenizer('spacy'))
inputs.build_vocab(utterances)

emb_dim = 100
inputs.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=emb_dim))
numerized_inputs, seq_len = inputs.process(utterances, device=-1, train=True)

"""--------------------------------------------------"""
batch_sz = 2
epochs = 1
vocab_sz = len(inputs.vocab)
model = sd.nnmodels.GRUClassifier(100, 10, vocab_sz, inputs.vocab.vectors,
                                  2, batch_sz)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

cnt = 0
start_time = time.time()
for epoch in range(epochs):
    for start in range(0, len(utterances), batch_sz):
        cnt += 1
        if cnt%100 == 0:
            eta = ((time.time()-start_time)/cnt)*(epochs*len(utterances)-cnt)/60
            eta_m = np.floor(eta)
            eta_s = (eta - eta_m) * 60
            perc = (cnt/(epochs*len(utterances)))*100
            out_str = "Progress: {:0.2f}%, ETA: {:0.0f}m{:0.0f}s".format(
                                                            perc, eta_m, eta_s)
            logging.info(out_str)

        model.zero_grad()

        model.hidden = model.init_hidden()

        sentence_in = numerized_inputs[start:start + batch_sz]
        targets = torch.tensor(labels[start:start + batch_sz], dtype=torch.long)

        try:
            log_scores = model(sentence_in)
            loss = loss_function(log_scores, targets)
            loss.backward()
            optimizer.step()
        except e:
            logging.warn(e)

with torch.no_grad():
    scores = np.exp(model(numerized_inputs[:2]))
    logging.info(scores)
