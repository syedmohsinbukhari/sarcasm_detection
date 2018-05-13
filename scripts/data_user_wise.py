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

fdname = 'data/main/final_data.json'
funame = 'data/main/user_wise.json'

big_dict = {}

with open(fdname) as f_data, open(funame, 'wb') as f_user:
    for line in f_data:
        logging.info("")

logging.info("Finished script data_user_wise.py")
