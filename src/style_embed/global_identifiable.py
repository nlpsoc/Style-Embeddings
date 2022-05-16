"""
    global constants (e.g., folders on our used cluster etc.)
        --> includes identifiable data, depends on local folder structures and cluster setup
        potentially needs to be changed to work with own data structure
"""

import sys
import os
import logging
from pathlib import Path

# project base folder
cur_dir = os.path.dirname(os.path.realpath(__file__))
BASE_FOLDER = cur_dir + "/../"
LOCAL_HOME = str(Path.home())
STRANFORMERS_CACHE = LOCAL_HOME + "/sentence_transformer/"  # "/home/USER/sentence_transformer/" for linux
OUTPUT_FOLDER = BASE_FOLDER + "output"
RESULTS_FOLDER = BASE_FOLDER + "output"

ON_HPC = False
if "uu_cs_nlpsoc" in BASE_FOLDER:  # for our local setup this checks if we are on HPC cluster,
    # file structures are different there, can probably be removed for your own setup
    ON_HPC = True

TRAIN_DATA_BASE = ""
CONVO_CACHE = LOCAL_HOME + "/convo-test"  # "/home/USER/convo-test for linux


def set_cache():
    logging.info('setting cache to cluster ')
    try:
        os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE
    except NameError:
        logging.info("using default cache for transformers")
    try:
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = STRANFORMERS_CACHE
    except NameError:
        logging.info("using default cache for sentence_transformers")


def include_STEL_project():
    # include the STEL project files if they are on the same level as the Style-Embeddings folder
    #   if this is not the case for your project you need to change the path '../../../STEL' to where you saved STEL
    cwd = os.getcwd()
    global STEL_PATH_DIM
    if 'utility' not in cwd:
        sys.path.append(os.path.join('../../../STEL', 'src'))
        sys.path.append(os.path.join('../../../STEL/src', 'utility'))
        STEL_PATH_DIM = [os.path.dirname(os.path.realpath(__file__)) +
                 '/../../../STEL/Data/STEL/dimensions/_quad_stel-dimensions_formal-815_complex-815.tsv']
    else:
        sys.path.append(os.path.join('../../../../STEL', 'src'))
        sys.path.append(os.path.join('../../../../STEL/src', 'utility'))
        STEL_PATH_DIM = [os.path.dirname(os.path.realpath(__file__)) +
                 '/../../../../STEL/Data/STEL/dimensions/_quad_stel-dimensions_formal-815_complex-815.tsv']

# LOCAL setup relict, can be removed for your own setup
if ON_HPC:
    BASE_FOLDER = "/hpc/uu_cs_nlpsoc/02-awegmann/"
    CONVO_CACHE = BASE_FOLDER + 'cache/convokit'
    TRAIN_DATA_BASE = BASE_FOLDER + 'style_discovery/train_data'
    STRANFORMERS_CACHE = BASE_FOLDER + 'sentence_transformers'
    TRANSFORMERS_CACHE = "/hpc/uu_cs_nlpsoc/02-awegmann/huggingface"


