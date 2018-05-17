# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:00:00 2018

@author: elcid
"""

""" Setup logging and environment """
# simulate that sarcasmdetection is installed as a python package
import context

#%%-----------------------------------------------------------------------------
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
import pickle

import sarcasmdetection as sd

#%%-----------------------------------------------------------------------------
sd.utils.setup_logging('logs/train_classifier.log')
logging.info("Running script train_classifier.py")

#%%-----------------------------------------------------------------------------
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
all_utterances = utterances
all_labels = labels

logging.info("Shuffling utterances")
np.random.seed(123)
indices = np.array(list(range(len(all_utterances))))
np.random.shuffle(indices)
all_utterances = [all_utterances[i] for i in indices
all_labels = [all_labels[i] for i in indices]

utterances = all_utterances[0:80000]
labels = all_labels[0:80000]

val_utterances = all_utterances[-2000:-1000]
val_labels = all_labels[-2000:-1000]
val_len = len(val_utterances)

test_utterances = all_utterances[-1000:]
test_labels = all_labels[-1000:]
test_len= len(test_utterances)
test_indices = indices[-1000:]

#%%-----------------------------------------------------------------------------
inputs = torchtext.data.Field(lower=True, include_lengths= True,
                              batch_first=True,
                              tokenize=torchtext.data.get_tokenizer('spacy'))
inputs.build_vocab(all_utterances)

emb_dim = 100
inputs.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=emb_dim))
numerized_inputs, seq_len = inputs.process(utterances, device=-1, train=False)

val_numerized_inputs, seq_len_val = inputs.process(val_utterances,
                                                        device=-1, train=False)
test_numerized_inputs, seq_len_test = inputs.process(test_utterances,
                                                        device=-1, train=False)

#%%-----------------------------------------------------------------------------
def infer_accuracy(model, labels, numerized_inputs, seq_len):
    with torch.no_grad():
        log_scores, hidden_out = model(numerized_inputs, seq_len)
        scores = np.exp(log_scores)
        pred_labels = np.argmax(scores.numpy(), axis=1)
        test_labels = np.array(labels)

        accuracy = sd.utils.compute_accuracy(pred_labels, test_labels)
        return accuracy, hidden_out.cpu().detach().numpy()

#%%-----------------------------------------------------------------------------
torch.device("cuda")

batch_sz = 1000
epochs = 5
word_emd_sz = 100
disp_size = 10
vocab_sz = len(inputs.vocab)
model = sd.nnmodels.GRUClassifier(100, word_emd_sz, vocab_sz,
                                  inputs.vocab.vectors, 2)

loss_function = nn.NLLLoss()
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []
train_accs = []
val_accs = []
early_stopping_cnt = 0
cnt = 0
loss = float('inf')
start_time = time.time()
for epoch in range(epochs):
    flg=False
    for start in range(0, len(utterances), batch_sz):
        cnt += 1

        model.zero_grad()

        model.hidden = model.init_hidden(batch_sz)

        if start+batch_sz > len(utterances):
            break
        sentence_in = numerized_inputs[start:start + batch_sz]
        targets = torch.tensor(labels[start:start + batch_sz],
                               dtype=torch.long).cuda()
        len_in = seq_len[start:start + batch_sz]

        log_scores, _ = model(sentence_in, len_in)
        loss = loss_function(log_scores, targets)
        loss.backward()
        optimizer.step()

        model.hidden = model.init_hidden(val_len)
        val_log_scores, _ = model(val_numerized_inputs, seq_len_val)
        val_loss = loss_function(val_log_scores, torch.tensor(val_labels,
                                 dtype=torch.long).cuda())

        train_losses.append(float(loss.cpu().detach().numpy()))
        val_losses.append(float(val_loss.cpu().detach().numpy()))

        if cnt%disp_size == 0:
            eta = ((time.time()-start_time)/(cnt*batch_sz))*\
                                    (epochs*len(utterances)-(cnt*batch_sz))/60
            eta_m = np.floor(eta)
            eta_s = (eta - eta_m) * 60
            perc = ((cnt*batch_sz)/(epochs*len(utterances)))*100
            out_str = "Progress: {:0.2f}%, ETA: {:0.0f}m{:0.0f}s".format(
                                                            perc, eta_m, eta_s)

            logging.info("Epoch: {0}".format(epoch))
            logging.info(out_str)
            logging.info("loss: "+str(train_losses[-1]))
            logging.info("val_loss: "+str(val_losses[-1]))

            train_acc, _ = infer_accuracy(model,
                                       labels[start:start + batch_sz],
                                       sentence_in, len_in)
            logging.info("Training accuracy {0}".format(train_acc))
            train_accs.append(train_acc)

            val_acc, _ = infer_accuracy(model, val_labels, val_numerized_inputs,
                                     seq_len_val)
            logging.info("Validation accuracy {0}".format(val_acc))
            val_accs.append(val_acc)

            if val_acc > 90 and epoch > 0:
                early_stopping_cnt += 1
                disp_str = "Incrementing early stopping counter to {0}"
                logging.info(disp_str.format(early_stopping_cnt))
            else:
                if early_stopping_cnt > 0:
                    logging.info("Resetting early stop counter")
                early_stopping_cnt = 0

            if early_stopping_cnt > 2:
                logging.info("Early stopping detected")
                break

    if early_stopping_cnt > 2:
        logging.info("Early stopping breaking the epoch loop")
        break

#%%-----------------------------------------------------------------------------
accuracy, test_embds = infer_accuracy(model, test_labels, test_numerized_inputs,
                                      seq_len_test)
logging.info("Test accuracy {0}".format(accuracy))

torch.save(model, 'data/models/' + model.__label__ + '.dat')
pickle.dump(test_embds, open('data/models/test_embeddings.pkl', 'wb'))
pickle.dump(test_labels, open('data/models/test_labels.pkl', 'wb'))

#%%-----------------------------------------------------------------------------
stats = {
    'train_accuracies': train_accs,
    'val_accuracies': val_accs,
    'train_losses': train_losses,
    'val_losses': val_losses
}

curr_time = str(round(time.time()))
stats_path = 'data/output/'+curr_time+'.json'
test_ind_path = 'data/output/test_indices/'+curr_time+'_test.json'
with open(stats_path, 'w') as f_stats, open(test_ind_path, 'w') as f_test:
    json.dump(stats, f_stats, indent=4, separators=(',', ': '))
    json.dump(test_indices.tolist(), f_test, indent=4, separators=(',', ': '))

logging.info("Finished script train_classifier.py")
