import os
import logging.config
import yaml


def setup_logging(default_level=logging.INFO):
    """
        Setup logging configuration
    """
    path = ""
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
