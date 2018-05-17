# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:00:00 2018

@author: arkhalid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

class GRUClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, embedding_vectors,
                 output_classes):
        super(GRUClassifier, self).__init__()
        self.__label__ = 'GRUClassifier'

        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data = embedding_vectors

        # The GRU takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim,
                          batch_first=True).cuda()

        # The linear layer that maps from hidden state space to tag space
        self.hidden2class = nn.Linear(hidden_dim, output_classes).cuda()

    def init_hidden(self, minibatchsize):
        return (torch.zeros(1, minibatchsize, self.hidden_dim).cuda())

    def forward(self, sentence, sent_len):
        embeds = self.word_embeddings(sentence.cuda())
        lens, indices = torch.sort(sent_len, 0, descending=True)
        pp_seq = nn.utils.rnn.pack_padded_sequence(embeds[indices],
                                                   lens.tolist(),
                                                   batch_first=True)
        minibatchsize = len(sentence)
        self.hidden = self.init_hidden(minibatchsize)
        rnn_out, self.hidden = self.rnn(pp_seq.cuda(), self.hidden)
        rnn_out,_ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True,
                                                   padding_value=0.0,
                                                   total_length=None)

        _, _indices = torch.sort(indices, 0)
        rnn_out = rnn_out[_indices]

        logits = self.hidden2class(self.hidden[-1])
        log_scores = F.log_softmax(logits, dim=1)
        return log_scores, self.hidden[-1]
