# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:00:00 2018

@author: elcid
"""

""" Setup logging and environment """
# simulate that sarcasmdetection is installed as a python package
import context

"""--------------------------------------------------"""

import logging
from sarcasmdetection.utils import setup_logging

setup_logging('logs/gru_classification.log')
logging.info("Running script gru_classification.py")

"""--------------------------------------------------"""

import sarcasmdetection as sd

"""--------------------------------------------------"""
import torch
import torchtext
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""--------------------------------------------------"""
utterances = ["mary had a little lamb", "mary had a sick lamb"]
labels = [0 , 1]

"""--------------------------------------------------"""
# use a tokenizer here for actual
train_data = [[word for word in utterance.split()] for utterance in utterances]

"""--------------------------------------------------"""
inputs = torchtext.data.Field(lower=True, include_lengths= True, batch_first=True)
inputs.build_vocab(train_data)

emb_dim = 100
inputs.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=emb_dim))
numerized_inputs, seq_len = inputs.process(train_data, device=-1, train=True)

logging.info(inputs.vocab.freqs)
logging.info(inputs.vocab.stoi)
logging.info(numerized_inputs)
logging.info(seq_len)

"""--------------------------------------------------"""
embedding = nn.Embedding(len(inputs.vocab), emb_dim)
embedding.weight.data = inputs.vocab.vectors
embedded_train_data = embedding(numerized_inputs)
logging.info(embedded_train_data.size())

"""--------------------------------------------------"""

batch_sz = 2
epochs = 300 # again, normally you would NOT do 300 epochs, it is toy data
vocab_sz = len(inputs.vocab)
model = sd.nnmodels.GRUClassifier(100, 10, vocab_sz, inputs.vocab.vectors, 2, batch_sz)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

with torch.no_grad():
    scores = model(numerized_inputs[:2])
    logging.info(scores)

for epoch in range(epochs):
    for start in range(0, len(utterances), batch_sz):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the GRU,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = numerized_inputs[start:start + batch_sz]
        targets = torch.tensor(labels[start:start + batch_sz], dtype=torch.long)
        # Step 3. Run our forward pass.
        log_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(log_scores, targets)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    scores = model(numerized_inputs[:2])
    logging.info(scores)
