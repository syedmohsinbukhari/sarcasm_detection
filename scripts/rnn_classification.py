# -*- coding: utf-8 -*-
"""
Created on Sun May  7 12:00:00 2018

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

# one way to import
from sarcasmdetection.nnmodels import TestClass
tco = TestClass()
tco.log_something("hello world")

# another way to import
import sarcasmdetection as sd
tco = sd.nnmodels.TestClass()
tco.log_something("this is a test script")
