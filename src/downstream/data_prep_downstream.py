import pandas as pd
import logging
from train_fns.config import Config
from common.log import setup_logging
import traceback

logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None


class DataPrepDownstream():

    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode

    def get_concat_lstm_data(self, feature_matrix):

        for i in range(feature_matrix.shape[0]):

            track = feature_matrix[i, :]
            track = track.reshape((1, len(track)))

            flag = 0
            for cut_seq_id in range(track.shape[1] // self.cfg.cut_seq_len):
                track_cut = track[:, cut_seq_id * self.cfg.cut_seq_len:(cut_seq_id + 1) * self.cfg.cut_seq_len]

                try:
                    if self.mode == 'train':
                        yield track_cut
                    else:
                        if cut_seq_id == track.shape[1] // self.cfg.cut_seq_len - 1:
                            flag = 1

                        yield track_cut, flag
                except Exception as e:
                    logger.error(traceback.format_exc())


if __name__ == '__main__':
    setup_logging()

    cfg = Config()
    data_ob_gene = DataPrepDownstream(cfg, mode='train')
