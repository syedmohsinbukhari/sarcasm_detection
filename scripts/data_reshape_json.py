# -*- coding: utf-8 -*-
"""
Created on Fri May 11 00:00:00 2018

@author: elcid
"""

""" Setup logging and environment """
# simulate that sarcasmdetection is installed as a python package
import context

#%%-----------------------------------------------------------------------------
import logging
from sarcasmdetection.utils import setup_logging

setup_logging('logs/data_reshape_json.log')
logging.info("Running script data_reshape_json.py")

#%%-----------------------------------------------------------------------------
import json
import os
import sys

with open('data/main/comments.json') as f_cmnts,\
     open('data/main/comments_new.json', 'w') as f_cmnts_new:
    cnt = 0
    lvl = 0
    item_cnt = 0
    flg = False
    buf = '{'
    in_quotes = 0
    while True:
        cnt += 1
        chr = f_cmnts.read(1)

        if lvl > 0:
            if lvl == 1:
                if chr == ',':
                    buf += '}'
                    buf += os.linesep
                    f_cmnts_new.write(buf)
                    buf = '{'
                    item_cnt += 1
                    flg = True
                elif chr != ' ':
                    buf += chr
            else:
                buf += chr

        if in_quotes >= 1:
            if in_quotes == 1:
                if chr == '\\':
                    in_quotes = 2
                elif chr == '"':
                    in_quotes = 0
            elif in_quotes == 2:
                in_quotes = 1
        else:
            if chr == '{':
                lvl += 1
            elif chr == '}':
                lvl -= 1
            elif chr == '"':
                in_quotes = 1

        if ((item_cnt%100000)==0) and flg:
            flg = False
            logging.info("Processed {0} items".format(item_cnt))

        if chr == '':
            break

    logging.info("Processed {0} items".format(item_cnt))
    logging.info("Script data_reshape_json.py ended")
