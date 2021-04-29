import numpy
import pandas
import os
import re
from os import listdir
import yaml
import traceback
import logging
from collections import namedtuple
from os.path import isfile, join

logger = logging.getLogger(__name__)


def load_config_as_class(model_dir, config_path, encoder_path, decoder_path, result_path):
    config = None
    try:
        with open(config_path, 'r') as infile:
            config = yaml.safe_load(infile)
            config['model_dir'] = model_dir
            config['encoder_path'] = encoder_path
            config['decoder_path'] = decoder_path
            config['config_save_file'] = config_path
            config['result_path'] = result_path
            config = namedtuple('config', config.keys())(*config.values())
    except:
        logger.error(traceback.format_exc())

    return config
