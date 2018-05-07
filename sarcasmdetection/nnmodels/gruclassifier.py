# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:00:00 2018

@author: elcid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, embedding_vectors,
                 output_classes, minibatch_size):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data = embedding_vectors
        self.minibatch_size = minibatch_size

        # The GRU takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim,
                          batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2class = nn.Linear(hidden_dim, output_classes)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, self.minibatch_size, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        rnn_out, self.hidden = self.rnn(embeds, self.hidden)

        logits = self.hidden2class(rnn_out[:,-1,:].squeeze_()) # verify this
        log_scores = F.log_softmax(logits, dim=1)
        return log_scores
