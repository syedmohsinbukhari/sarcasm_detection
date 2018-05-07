# -*- coding: utf-8 -*-
"""
Created on Sun May  7 12:00:00 2018

@author: arkhalid
"""

""" Setup logging and environment """
# simulate that sarcasmdetection is installed as a python package
import context

import logging

def setup_logging(log_fname):
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler("{0}".format(log_fname))
    fileFormatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s " +
                                      "- %(funcName)s() ] [%(levelname)s] " +
                                      "%(message)s")
    fileHandler.setFormatter(fileFormatter)
    fileHandler.setLevel(logging.INFO)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleFormatter = logging.Formatter("[%(levelname)s]  %(message)s")
    consoleHandler.setFormatter(consoleFormatter)
    consoleHandler.setLevel(logging.INFO)
    rootLogger.addHandler(consoleHandler)

setup_logging('logs/pytorch_test.log')
logging.info("Running script pytorch_test.py")

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
batch_size = 2
train_data = [[word for word in utterance.split()] for utterance in utterances]
logging.info(train_data)

"""--------------------------------------------------"""
inputs = torchtext.data.Field(lower=True, include_lengths= True, batch_first=True)
inputs.build_vocab(train_data)

emb_dim = 100
inputs.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=emb_dim))
logging.info(inputs.vocab.freqs)
logging.info(inputs.vocab.stoi)
numericalized_inputs, seq_len = inputs.process(train_data, device=-1, train=True)
logging.info(numericalized_inputs)
logging.info(seq_len)

"""--------------------------------------------------"""
embedding = nn.Embedding(len(inputs.vocab), emb_dim)
embedding.weight.data = inputs.vocab.vectors
embedded_train_data = embedding(numericalized_inputs)
logging.info(embedded_train_data.size())

"""--------------------------------------------------"""
# cell = nn.GRU(input_size=100,hidden_size=10, batch_first=True)

# # initialize the hidden state.
# hidden = (torch.randn(1, 2, 10))
# out, hidden = cell(embedded_train_data, hidden)

class SarcasmModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, embedding_vectors, output_classes, minibatch_size):
        super(SarcasmModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight.data = embedding_vectors
        self.minibatch_size = minibatch_size
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.rnn = nn.GRU(input_size=embedding_dim,hidden_size=hidden_dim, batch_first=True)

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
        #[B, sent_len] - >  [B, sent_len, emb_dim]
        embeds = self.word_embeddings(sentence)
        # [B, sent_len, hid_dim]
        rnn_out, self.hidden = self.rnn(
           embeds, self.hidden)
        #print(self.hidden.size())
        #print(rnn_out.size)
        logits = self.hidden2class(rnn_out[:,-1,:].squeeze_()) # verify this
        log_scores = F.log_softmax(logits, dim=1)
        return log_scores

"""--------------------------------------------------"""
epochs = 300
model = SarcasmModel(100, 10, len(inputs.vocab), inputs.vocab.vectors, 2, batch_size)
#print(model.forward(numericalized_inputs))

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

with torch.no_grad():
    scores = model(numericalized_inputs[:2])
    logging.info(scores)

for epoch in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    for start in range(0, len(utterances), batch_size):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = numericalized_inputs[start:start + batch_size]
        targets = torch.tensor(labels[start:start + batch_size], dtype=torch.long)
        # Step 3. Run our forward pass.
        log_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(log_scores, targets)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    scores = model(numericalized_inputs[:2])
    logging.info(scores)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
