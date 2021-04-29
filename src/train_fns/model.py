from __future__ import division
from torch import optim
import torch
from torch import nn
from train_fns import encoder, decoder
from keras.optimizers import Adam
from keras.layers import Input
import logging

logger = logging.getLogger(__name__)


class Model:

    def __init__(self, cfg, vocab_size, gpu_id):
        self.cfg = cfg
        self.encoder = encoder.EncoderLSTM(cfg, cfg.input_size_encoder, cfg.hidden_size_encoder, gpu_id).cuda(
            gpu_id).train()
        self.decoder = decoder.DecoderLSTM(cfg, cfg.input_size_decoder, cfg.hidden_size_decoder,
                                           cfg.output_size_decoder, gpu_id).cuda(gpu_id).train()

    def load_weights(self):
        try:
            print('loading weights from {}'.format(self.cfg.model_dir))
            self.encoder.load_state_dict(torch.load(self.cfg.model_dir + '/encoder.pth'))
            self.decoder.load_state_dict(torch.load(self.cfg.model_dir + '/decoder.pth'))
            # self.ca_embedding.load_state_dict(torch.load(self.cfg.model_dir + '/ca_embedding.pth'))
        except Exception as e:
            print("load weights exception: {}".format(e))

    def compile_optimizer(self):
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.cfg.learning_rate)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.cfg.learning_rate)
        criterion = nn.MSELoss()
        return encoder_optimizer, decoder_optimizer, criterion

    def set_callback(self, callback):
        callback.set_model(self.encoder)
        callback.set_model(self.decoder)
        pass

    def save_weights(self):
        torch.save(self.encoder.state_dict(), self.cfg.model_dir + '/encoder.pth')
        torch.save(self.decoder.state_dict(), self.cfg.model_dir + '/decoder.pth')
        pass
