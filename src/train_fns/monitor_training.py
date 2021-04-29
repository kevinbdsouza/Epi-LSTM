from __future__ import division
import numpy as np
from collections import Counter
from keras.utils import generic_utils
import tensorflow as tf
import yaml
import logging
import traceback
import pandas as pd

logger = logging.getLogger(__name__)


class MonitorTraining:

    def __init__(self, cfg, vocab_size):
        self.best_loss = np.Inf
        self.cfg = cfg
        self.losses_iter = []
        self.losses_epoch = []

    @staticmethod
    def save_config_as_yaml(path, cfg):
        """
            Save configuration
        """
        try:
            with open(path, 'w') as f:
                yaml.safe_dump(cfg.__dict__, f, default_flow_style=False)
        except:
            logger.error(traceback.format_exc())

    def add_tf_summary(self, callback, loss_dict, seq_num):
        for name, value in loss_dict.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, seq_num)
            callback.writer.flush()

    def monitor_loss_iter(self, callback, rec_loss, iter_num):
        self.losses_iter.append(rec_loss)
        loss_dict_iter = {'rec_loss_iter': rec_loss}
        self.add_tf_summary(callback, loss_dict_iter, iter_num)

    def monitor_loss_epoch(self, callback, rec_loss, iter_num, epoch_num):

        epoch_loss = np.mean(self.losses_iter)
        self.losses_iter = []

        self.losses_epoch.append(epoch_loss)
        loss_dict_epoch = {'rec_loss_epoch': epoch_loss}
        self.add_tf_summary(callback, loss_dict_epoch, epoch_num)

        print('Mean reconstruction loss at end of epoch for all epgen: {}'.format(epoch_loss))

        if epoch_loss < self.best_loss:
            print('Total loss decreased from {} to {}, saving weights'.format(self.best_loss, rec_loss))
            self.best_loss = rec_loss
            return 'True'
        else:
            return 'False'
