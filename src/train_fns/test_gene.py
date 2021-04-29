from __future__ import division
from train_fns.data_prep_gene import DataPrepGene
from train_fns.monitor_testing import MonitorTesting
from train_fns.train_gene import unroll_loop
from eda.viz import Viz
from train_fns.model import Model
import logging
from common.log import setup_logging
from common import utils
import traceback
import torch
from torch.autograd import Variable
from keras.callbacks import TensorBoard
import numpy as np
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpu_id = 0
mode = "test"

logger = logging.getLogger(__name__)


def get_config(model_dir, config_base, result_base):
    encoder_path = os.path.join(model_dir, '/encoder.pth')
    decoder_path = os.path.join(model_dir, '/decoder.pth')
    config_path = os.path.join(model_dir, config_base)
    res_path = os.path.join(model_dir, result_base)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    cfg = utils.load_config_as_class(model_dir, config_path, encoder_path, decoder_path, res_path)
    return cfg


def test_gene(cfg):
    data_ob_gene = DataPrepGene(cfg, mode='test')
    monitor = MonitorTesting(cfg)
    callback = TensorBoard(cfg.tensorboard_log_path)
    vizOb = Viz(cfg)

    data_ob_gene.prepare_id_dict()
    data_gen_test = data_ob_gene.get_data()
    model = Model(cfg, data_ob_gene.vocab_size, gpu_id)
    model.load_weights()
    model.set_callback(callback)

    logger.info('Testing Start')

    iter_num = 0
    hidden_states = np.zeros((2, cfg.hidden_size_encoder))
    # hidden_states = []
    encoder_init = True
    decoder_init = True
    encoder_optimizer, decoder_optimizer, criterion = None, None, None

    try:
        for track_cut in data_gen_test:

            mse, hidden_states, encoder_init, decoder_init, predicted_cut = unroll_loop(cfg, track_cut, model,
                                                                                        encoder_optimizer,
                                                                                        decoder_optimizer, criterion,
                                                                                        hidden_states, encoder_init,
                                                                                        decoder_init, mode)

            iter_num += 1
            if iter_num % 500 == 0:
                logger.info('Iter: {} - mse: {}'.format(iter_num, np.mean(monitor.mse_iter)))
                vizOb.plot_prediction(predicted_cut, track_cut, mse, iter_num)

            monitor.monitor_mse_iter(callback, np.sum(mse), iter_num)

    except Exception as e:
        logger.error(traceback.format_exc())

    logging.info('Testing complete')
    print('Mean MSE at end of testing: {}'.format(np.mean(monitor.mse_iter)))


if __name__ == '__main__':
    setup_logging()

    model_dir = '..data/'
    config_base = 'config.yaml'
    result_base = 'images'

    cfg = get_config(model_dir, config_base, result_base)

    test_gene(cfg)
