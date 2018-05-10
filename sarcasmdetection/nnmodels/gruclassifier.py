# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:00:00 2018

@author: elcid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

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
                          batch_first=True).cuda()

        # The linear layer that maps from hidden state space to tag space
        self.hidden2hidden = nn.Linear(hidden_dim, output_classes*10).cuda()
        self.hidden2class = nn.Linear(output_classes*10, output_classes).cuda()

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, self.minibatch_size, self.hidden_dim).cuda())

    def forward(self, sentence, sent_len):
        embeds = self.word_embeddings(sentence)
        lens, indices = torch.sort(sent_len, 0, descending=True)
        pp_seq = nn.utils.rnn.pack_padded_sequence(embeds[indices],
                                                   lens.tolist(),
                                                   batch_first=True)
        rnn_out, self.hidden = self.rnn(pp_seq.cuda(), self.hidden)
        rnn_out,_ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True,
                                                   padding_value=0.0,
                                                   total_length=None)

        _, _indices = torch.sort(indices, 0)
        rnn_out = rnn_out[_indices]

        hidden_out = self.hidden2hidden(rnn_out[:,-1,:].squeeze_())
        hidden_out = self.hidden2hidden(self.hidden[-1])
        hidden_out_act = F.sigmoid(hidden_out)

        logits = self.hidden2class(hidden_out_act)
        log_scores = F.log_softmax(logits, dim=1)
        return log_scores
