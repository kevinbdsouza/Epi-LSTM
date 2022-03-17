import logging
import numpy as np
import pandas as pd
from downstream import config_downstream
from train_fns.model import Model
from train_fns.monitor_training import MonitorTraining
from keras.callbacks import TensorBoard
from train_fns.train_gene import unroll_loop
import traceback
from train_fns.monitor_testing import MonitorTesting
from eda.viz import Viz
from downstream.data_prep_downstream import DataPrepDownstream
from train_fns.test_gene import get_config

logger = logging.getLogger(__name__)


class DownstreamLSTM:
    def __init__(self):
        self.gpu_id = 0
        self.mode = 'train'
        self.run_train = False
        self.config_base = 'config_down.yaml'
        self.result_base = 'images'

    def get_features(self, feature_matrix, target):
        new_feature_matrix = pd.DataFrame()

        cfg = config_downstream.ConfigDownstream()

        if self.run_train:
            model = self.train_down_lstm(feature_matrix, cfg)
        else:
            model = Model(cfg, cfg.input_size_encoder, self.gpu_id)
            model.load_weights()

            cfg = get_config(cfg.model_dir, self.config_base, self.result_base)

        features = self.test_down_lstm(feature_matrix, model, cfg)

        for i in range(feature_matrix.shape[0]):
            feat_row = np.append(features, target[i], axis=1)
            new_feature_matrix = new_feature_matrix.append(pd.DataFrame(feat_row),
                                                           ignore_index=True)

        return new_feature_matrix, cfg

    def train_down_lstm(self, feature_matrix, cfg):

        model = Model(cfg, cfg.input_size_encoder, self.gpu_id)
        data_ob_down = DataPrepDownstream(cfg, mode='train')

        if cfg.load_weights:
            model.load_weights()

        monitor = MonitorTraining(cfg, cfg.input_size_encoder)
        callback = TensorBoard(cfg.tensorboard_log_path)
        monitor.save_config_as_yaml(cfg.config_file, cfg)

        encoder_optimizer, decoder_optimizer, criterion = model.compile_optimizer()
        model.set_callback(callback)

        for epoch_num in range(cfg.num_epochs):

            logger.info('Epoch {}/{}'.format(epoch_num + 1, cfg.num_epochs))

            data_gen_train = data_ob_down.get_concat_lstm_data(feature_matrix)

            iter_num = 0
            hidden_states = np.zeros((2, cfg.hidden_size_encoder))
            rec_loss = 0
            encoder_init = True
            decoder_init = True

            try:
                for track_cut in data_gen_train:

                    rec_loss, hidden_states, encoder_init, decoder_init, predicted_cut, encoder_hidden_states_np = unroll_loop(
                        cfg, track_cut,
                        model,
                        encoder_optimizer,
                        decoder_optimizer,
                        criterion,
                        hidden_states,
                        encoder_init,
                        decoder_init, self.mode)

                    iter_num += 1
                    if iter_num % 500 == 0:
                        logger.info('Iter: {} - rec_loss: {}'.format(iter_num, np.mean(monitor.losses_iter)))
                        model.save_weights()

                    monitor.monitor_loss_iter(callback, rec_loss, iter_num)

                save_flag = monitor.monitor_loss_epoch(callback, rec_loss, iter_num, epoch_num)
                if save_flag == 'True':
                    model.save_weights()

            except Exception as e:
                logger.error(traceback.format_exc())
                model.save_weights()
                continue

        model.save_weights()
        logging.info('Training complete, exiting.')

        return model

    def test_down_lstm(self, feature_matrix, model, cfg):

        self.mode = 'test'

        monitor = MonitorTesting(cfg)
        data_ob_down = DataPrepDownstream(cfg, mode='test')
        data_gen_test = data_ob_down.get_concat_lstm_data(feature_matrix)

        callback = TensorBoard(cfg.tensorboard_log_path)
        vizOb = Viz(cfg)

        model.set_callback(callback)

        logger.info('Testing Start')

        iter_num = 0
        count = 0
        hidden_states = np.zeros((2, cfg.hidden_size_encoder))
        gene_states = np.zeros((feature_matrix.shape[0]), cfg.hidden_size_encoder)
        encoder_init = True
        decoder_init = True
        encoder_optimizer, decoder_optimizer, criterion = None, None, None

        try:
            for track_cut, flag in data_gen_test:

                mse, hidden_states, encoder_init, decoder_init, predicted_cut, encoder_hidden_states_np = unroll_loop(
                    cfg, track_cut,
                    model,
                    encoder_optimizer,
                    decoder_optimizer,
                    criterion,
                    hidden_states, encoder_init,
                    decoder_init, self.mode)

                if flag == 1:
                    gene_states[count, :] = hidden_states[0]
                    count += 1

                iter_num += 1
                if iter_num % 500 == 0:
                    logger.info('Iter: {} - mse: {}'.format(iter_num, np.mean(monitor.mse_iter)))

                monitor.monitor_mse_iter(callback, np.sum(mse), iter_num)

        except Exception as e:
            logger.error(traceback.format_exc())

        logging.info('Testing complete')
        print('Mean MSE at end of testing: {}'.format(np.mean(monitor.mse_iter)))

        return gene_states
