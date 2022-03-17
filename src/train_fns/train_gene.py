from __future__ import division
from train_fns import config
from train_fns.data_prep_gene import DataPrepGene
from train_fns.monitor_training import MonitorTraining
from train_fns.model import Model
import logging
from common.log import setup_logging
import traceback
import pandas as pd
import torch
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_
from keras.callbacks import TensorBoard
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpu_id = 1
mode = 'train'

logger = logging.getLogger(__name__)


def train_iter_gene(cfg, chr):
    data_ob_gene = DataPrepGene(cfg, mode='train', chr=str(chr))

    data_ob_gene.prepare_id_dict()
    model = Model(cfg, data_ob_gene.vocab_size, gpu_id)
    
    monitor = MonitorTraining(cfg, data_ob_gene.vocab_size)
    callback = TensorBoard(cfg.tensorboard_log_path)
    monitor.save_config_as_yaml(cfg.config_file, cfg)

    encoder_optimizer, decoder_optimizer, criterion = model.compile_optimizer()
    model.set_callback(callback)

    for epoch_num in range(cfg.num_epochs):

        logger.info('Epoch {}/{}'.format(epoch_num + 1, cfg.num_epochs))

        data_gen_train = data_ob_gene.get_data()

        iter_num = 0
        hidden_states = np.zeros((2, cfg.hidden_size_encoder))
        # hidden_states = []
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
                    decoder_init, mode)

                # logger.info('Hidden states: {}'.format(encoder_hidden_states_np))

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


def unroll_loop(cfg, track_cut, model, encoder_optimizer,
                decoder_optimizer, criterion, hidden_states, encoder_init, decoder_init, mode):
    encoder = model.encoder
    decoder = model.decoder

    if encoder_init:
        encoder_hidden, encoder_state = encoder.initHidden()
        encoder_init = False
    else:
        encoder_hidden_init = hidden_states[0]
        # encoder_state_init = hidden_states[1]
        encoder_hidden = Variable(torch.from_numpy(encoder_hidden_init).float().unsqueeze(0).unsqueeze(0)).cuda(gpu_id)
        # encoder_state = Variable(torch.from_numpy(encoder_state_init).float().unsqueeze(0).unsqueeze(0)).cuda()
        _, encoder_state = encoder.initHidden()

    if mode == "train":
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

    input_variable = track_cut
    target_variable = track_cut
    nValues = target_variable.shape[1]

    encoder_outputs = Variable(torch.zeros(nValues, 1, cfg.hidden_size_encoder)).cuda(gpu_id)
    encoder_hidden_states = Variable(torch.zeros(nValues, 1, cfg.hidden_size_encoder)).cuda(gpu_id)
    encoder_states = Variable(torch.zeros(nValues, 1, cfg.hidden_size_encoder)).cuda(gpu_id)

    rec_loss = 0
    for ei in range(0, nValues):

        if not np.any(input_variable[:, ei]):
            continue

        encoder_input = Variable(
            torch.from_numpy(np.array(input_variable[:, ei])).float().unsqueeze(0).unsqueeze(0)).cuda(gpu_id)

        encoder_output, encoder_hidden, encoder_state = encoder(encoder_input, encoder_hidden, encoder_state)
        encoder_outputs[ei] = encoder_output[0][0]
        encoder_hidden_states[ei] = encoder_hidden[0][0]
        encoder_states[ei] = encoder_state[0][0]

    hidden_states[0] = encoder_hidden.squeeze(0).cpu().data.numpy()
    hidden_states[1] = encoder_state.squeeze(0).cpu().data.numpy()
    decoder_hidden = encoder_hidden

    predicted_cut = np.zeros((cfg.input_size_encoder, cfg.cut_seq_len))
    # With teacher forcing
    for di in range(0, nValues):
        decoder_state = encoder_states[di].unsqueeze(0)

        decoder_output, decoder_hidden, _ = decoder(encoder_outputs[di].unsqueeze(0), decoder_hidden,
                                                    decoder_state)

        if mode == "train":
            decoder_target = Variable(
                torch.from_numpy(np.array(target_variable[:, di])).float().unsqueeze(0)).cuda(gpu_id)
            rec_loss += criterion(decoder_output.squeeze(0), decoder_target)
        else:
            decoder_prediction = decoder_output.squeeze(0).cpu().data.numpy()
            predicted_cut[:, di] = decoder_prediction

            rec_loss += np.power((decoder_prediction - target_variable[:, di]), 2)

    if mode == "train":
        rec_loss.backward()

        clip_grad_norm_(encoder.parameters(), max_norm=cfg.max_norm)
        clip_grad_norm_(decoder.parameters(), max_norm=cfg.max_norm)

        encoder_optimizer.step()
        decoder_optimizer.step()
        mean_loss = rec_loss.item() / nValues
    else:
        mean_loss = rec_loss / nValues

    encoder_hidden_states_np = encoder_hidden_states.squeeze(1).cpu().data.numpy()

    return mean_loss, hidden_states, encoder_init, decoder_init, predicted_cut, encoder_hidden_states_np


if __name__ == '__main__':
    setup_logging()
    chr = '21'
    cfg = config.Config()
    train_iter_gene(cfg, chr)
