import logging

from os import listdir
from os.path import isfile, join

def setup_logging(log_fname=''):
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)

    if log_fname != '':
        format_string = "%(asctime)s [%(filename)s:%(lineno)s - "+\
                        "%(funcName)s() ] [%(levelname)s] %(message)s"
        fileHandler = logging.FileHandler("{0}".format(log_fname))
        fileFormatter = logging.Formatter(format_string)
        fileHandler.setFormatter(fileFormatter)
        fileHandler.setLevel(logging.INFO)
        rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleFormatter = logging.Formatter("[%(levelname)s]  %(message)s")
    consoleHandler.setFormatter(consoleFormatter)
    consoleHandler.setLevel(logging.INFO)
    rootLogger.addHandler(consoleHandler)

def compute_accuracy(true_labels, pred_labels):
    matches = 0
    for i in range(len(true_labels)):
        if true_labels[i] == pred_labels[i]:
            matches += 1
    accuracy = (matches/len(true_labels)) * 100

    return accuracy

def get_biggest_fname(inp_path):
    fnames = [x for x in listdir(inp_path) if isfile(join(inp_path, x))]
    fnames.sort()
    return join(inp_path, fnames[-1])
