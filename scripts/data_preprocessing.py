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

setup_logging('logs/test_script.log')
logging.info("Running script test_script.py")

"""--------------------------------------------------"""
import json
import os
import sys

with open('data/pol/comments.json') as f_cmnts, \
     open('data/pol/train-balanced.csv') as f_tb, \
     open('data/pol/final_data.json', 'w') as f_final:

    cmnts = json.load(f_cmnts)

    final_data = {}
    cnt = 0

    for line in f_tb:
        cnt += 1
        processed_line = line.replace('\n', '')
        processed_line = processed_line.replace('\r', '')
        all_fields = processed_line.split('|')
        resps = all_fields[1].split(' ')
        labels = all_fields[2].split(' ')

        for i in range(len(resps)):
            if cmnts[resps[i]]['author'] == "[deleted]":
                continue
            final_data[resps[i]] = {'text': cmnts[resps[i]]['text'],
                                   'author': cmnts[resps[i]]['author'],
                                   'label': labels[i]}

        if cnt%100 == 0:
            print('.', end='')
            sys.stdout.flush()

    json_dump = json.dumps(final_data)
    f_final.write(json_dump)

    print()

    logging.info(cnt)
