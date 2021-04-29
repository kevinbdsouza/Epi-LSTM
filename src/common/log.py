import os
import logging.config
from common.constants import Constants
import yaml


def setup_logging(default_level=logging.INFO):
    """
        Setup logging configuration
    """
    path = Constants.LOG_SETTINGS
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
