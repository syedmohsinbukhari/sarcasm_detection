# -*- coding: utf-8 -*-
"""
Created on Mon May  13 00:00:00 2018

@author: elcid
"""

""" Setup logging and environment """
# simulate that sarcasmdetection is installed as a python package
import context

#%%-----------------------------------------------------------------------------
import json
import matplotlib.pyplot as plt
import logging

import sarcasmdetection as sd

#%%-----------------------------------------------------------------------------
sd.utils.setup_logging('logs/plot_statistics.log')
logging.info("Running script plot_statistics.py")

#%%-----------------------------------------------------------------------------
fname = sd.utils.get_biggest_fname('data/output')

with open(fname) as f:
    stats = json.load(f)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(stats['train_losses'], label='train_losses')
    ax1.plot(stats['val_losses'], label='val_losses')
    ax2.plot(stats['train_accuracies'], label='train_accuracies')
    ax2.plot(stats['val_accuracies'], label='val_accuracies')

    ax1.set_title('Losses')
    ax2.set_title('Accuracies')

    ax1.legend()
    ax2.legend()

    plt.tight_layout()

    plt.show()

logging.info("Finished script plot_statistics.py")
