# -*- coding: utf-8 -*-
"""
Created on Mon May  7 12:00:00 2018

@author: elcid
"""

""" Setup logging and environment """
# simulate that sarcasmdetection is installed as a python package
import context

#%%-----------------------------------------------------------------------------
import logging
from sarcasmdetection.utils import setup_logging

setup_logging('logs/test_script.log')
logging.info("Running script test_script.py")

#%%-----------------------------------------------------------------------------
import json
import os
import sys

with open('data/main/comments_new.json') as f_cmnts, \
     open('data/main/train-balanced.csv') as f_tb, \
     open('data/main/final_data.json', 'w') as f_final:

    labels_dict = {}
    cnt = 0
    for line in f_tb:
        cnt += 1
        processed_line = line.replace('\n', '')
        processed_line = processed_line.replace('\r', '')
        all_fields = processed_line.split('|')
        resps = all_fields[1].split(' ')
        labels = all_fields[2].split(' ')

        for i in range(len(resps)):
            labels_dict[resps[i]] = {'label': labels[i]}

    cnt = 0
    cnt_labels = [0, 0]
    for line in f_cmnts:
        cmnt_dict = json.loads(line)
        cmnt_id = list(cmnt_dict.keys())[0]

        processed_dict = {}
        if cmnt_id in labels_dict.keys():
            if cmnt_dict[cmnt_id]['author']=='[deleted]':
                continue
            cnt += 1
            cmnt_label = labels_dict[cmnt_id]['label']
            if cmnt_label == '0':
                cnt_labels[0] = cnt_labels[0] + 1
            elif cmnt_label == '1':
                cnt_labels[1] = cnt_labels[1] + 1
            processed_dict[cmnt_id] = {
                'text': cmnt_dict[cmnt_id]['text'],
                'author': cmnt_dict[cmnt_id]['author'],
                'label': cmnt_label
            }
            json_dump = json.dumps(processed_dict)
            f_final.write(json_dump)
            f_final.write(os.linesep)

            if (cnt%10000)==0:
                logging.info('Processed {0} lines'.format(cnt))
        else:
            continue

    logging.info('Processed {0} lines'.format(cnt))
    logging.info('Sarcastic comments: {0}, non-sarcastic: {1}'.format(
                                                cnt_labels[0], cnt_labels[1]))
    logging.info("Script data_preprocessing.py ended")
