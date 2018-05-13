# -*- coding: utf-8 -*-
"""
Created on Fri May 11 00:00:00 2018

@author: elcid
"""

""" Setup logging and environment """
# simulate that sarcasmdetection is installed as a python package
import context

"""--------------------------------------------------"""
import logging
from sarcasmdetection.utils import setup_logging

setup_logging('logs/data_user_wise.log')
logging.info("Running script data_user_wise.py")

"""--------------------------------------------------"""
import json
import os

fdname = 'data/main/final_data.json'
funame = 'data/main/user_wise.json'

user_dict = {}

with open(fdname) as f_data, open(funame, 'w') as f_user:
    for line in f_data:
        cmnt_dict = json.loads(line)
        cmnt_id = list(cmnt_dict.keys())[0]
        cmnt_author = cmnt_dict[cmnt_id]['author']

        if not cmnt_author in user_dict.keys():
            user_dict[cmnt_author] = {'comment_count': 0}

        user_dict[cmnt_author][cmnt_id] = {
            'text': cmnt_dict[cmnt_id]['text'],
            'label': cmnt_dict[cmnt_id]['label']
        }
        user_dict[cmnt_author]['comment_count'] += 1

    json_dump = json.dumps(user_dict)
    f_user.write(json_dump)

logging.info("Finished script data_user_wise.py")
