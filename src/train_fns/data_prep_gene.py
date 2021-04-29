import pandas as pd
import logging
import re
from train_fns.config import Config
import numpy as np
from os import listdir
from os.path import isfile, join
from common.log import setup_logging
import traceback
import math

logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None


class DataPrepGene():

    def __init__(self, cfg, mode, chr):
        self.epigenome_dict = {}
        self.assay_dict = {}
        self.epigenome_assay_dict = {}
        self.cfg = cfg
        self.mode = mode
        self.chr_len = self.cfg.chr_len[chr]
        self.chr_cuts = self.cfg.chr_cuts[chr]
        self.ch_cut_len = int(math.floor((self.chr_len // self.chr_cuts) / 100.0)) * 100
        self.tracks = None
        self.epigenome_npz_path = cfg.epigenome_npz_path_train
        self.epigenome_bigwig_path = cfg.epigenome_bigwig_path
        self.fasta_path = cfg.fasta_path

    def get_data(self):

        # fasta_files = [f for f in listdir(self.fasta_path) if isfile(join(self.fasta_path, f))]
        # fasta_files.sort()

        max_len = 0

        last_pos = self.chr_cuts * self.ch_cut_len

        for num in range(self.chr_cuts):

            print("num {}".format(num))

            for epgen_assay, epgen_assay_id in sorted(self.epigenome_assay_dict.items()):
                full_track_path = self.epigenome_npz_path + "/" + epgen_assay + ".npz"
                track = np.load(full_track_path)
                track = track['arr_0'][0]

                if len(track) > max_len:
                    max_len = len(track)

                if num == self.chr_cuts - 1 and len(track) <= last_pos:
                    track_patch = track[num * self.ch_cut_len:len(track)]
                    self.tracks[epgen_assay_id][:len(track_patch)] = track_patch
                    self.tracks[epgen_assay_id][len(track_patch):self.ch_cut_len] = np.zeros(
                        (1, self.ch_cut_len - len(track_patch)))
                else:
                    track_patch = track[num * self.ch_cut_len:(num + 1) * self.ch_cut_len]
                    self.tracks[epgen_assay_id][:len(track_patch)] = track_patch

            for cut_seq_id in range(self.ch_cut_len // self.cfg.cut_seq_len):
                track_cut = self.tracks[:, cut_seq_id * self.cfg.cut_seq_len:(cut_seq_id + 1) * self.cfg.cut_seq_len]

                try:
                    yield track_cut
                except Exception as e:
                    logger.error(traceback.format_exc())

        for epgen_assay, epgen_assay_id in sorted(self.epigenome_assay_dict.items()):
            full_track_path = self.epigenome_npz_path + "/" + epgen_assay + ".npz"
            track = np.load(full_track_path)
            track = track['arr_0'][0]
            self.tracks[epgen_assay_id][:self.cfg.cut_seq_len] = np.zeros((1, self.cfg.cut_seq_len))

            if len(track) > last_pos:
                self.tracks[epgen_assay_id][:len(track) - last_pos] = track[last_pos:len(track)]

        try:
            yield self.tracks[:, :self.cfg.cut_seq_len]
        except Exception as e:
            logger.error(traceback.format_exc())

    def get_track_values(self, cut_seq_id, track):
        track_ids = list(
            set(list(range((cut_seq_id + 1) * self.cfg.cut_seq_len // self.cfg.base_pair_resolution))) - set(
                list(range(cut_seq_id * self.cfg.cut_seq_len // self.cfg.base_pair_resolution))))

        track_cut = track[track_ids]

        return track_cut

    def prepare_id_dict(self):
        npz_files = [f for f in listdir(self.epigenome_npz_path) if isfile(join(self.epigenome_npz_path, f))]
        npz_files.sort()
        epgen_id = 1
        assay_id = 1

        for id, file in enumerate(npz_files):
            epigenome_assay = re.split(r"\-\s*", re.split(r"\.\s*", file)[0])

            if epigenome_assay[0] not in self.epigenome_dict:
                self.epigenome_dict[epigenome_assay[0]] = epgen_id
                epgen_id += 1

            if epigenome_assay[1] not in self.assay_dict:
                self.assay_dict[epigenome_assay[1]] = assay_id
                assay_id += 1

            self.epigenome_assay_dict[epigenome_assay[0] + "-" + epigenome_assay[1]] = id

        self.vocab_size = len(self.epigenome_assay_dict)

        if self.mode == 'train':
            self.cfg.input_size_encoder = self.vocab_size
            self.cfg.output_size_decoder = self.vocab_size
        elif self.mode == 'test':
            self.cfg = self.cfg._replace(input_size_encoder=self.vocab_size)
            self.cfg = self.cfg._replace(output_size_decoder=self.vocab_size)

        self.inv_epgen_dict = {v: k for k, v in self.epigenome_assay_dict.items()}
        self.tracks = np.zeros((self.vocab_size, self.ch_cut_len))

    def get_concat_lstm_data(self, feature_matrix):

        for i in range(feature_matrix.shape[0]):

            track = feature_matrix[i, :]
            track = track.reshape((1, len(track)))

            for cut_seq_id in range(self.cfg.chr21_len // self.cfg.cut_seq_len):
                track_cut = track[:, cut_seq_id * self.cfg.cut_seq_len:(cut_seq_id + 1) * self.cfg.cut_seq_len]

                try:
                    yield track_cut
                except Exception as e:
                    logger.error(traceback.format_exc())

    def get_assay_data_at_positions(self, cfg, mask_vector, gene_ar):

        max_len = 0
        last_pos = self.chr_cuts * self.ch_cut_len
        pd_col = list(np.arange(cfg.input_size_encoder))
        pd_col.append('gene_id')
        assay_matrix = pd.DataFrame(columns=pd_col)

        for num in range(self.chr_cuts):

            print("num {}".format(num))

            for epgen_assay, epgen_assay_id in sorted(self.epigenome_assay_dict.items()):
                full_track_path = self.epigenome_npz_path + "/" + epgen_assay + ".npz"
                track = np.load(full_track_path)
                track = track['arr_0'][0]

                if len(track) > max_len:
                    max_len = len(track)

                if num == self.chr_cuts - 1 and len(track) <= last_pos:
                    track_patch = track[num * self.ch_cut_len:len(track)]
                    self.tracks[epgen_assay_id][:len(track_patch)] = track_patch
                    self.tracks[epgen_assay_id][len(track_patch):self.ch_cut_len] = np.zeros(
                        (1, self.ch_cut_len - len(track_patch)))
                else:
                    track_patch = track[num * self.ch_cut_len:(num + 1) * self.ch_cut_len]
                    self.tracks[epgen_assay_id][:len(track_patch)] = track_patch

            mask_vec_cut = mask_vector[num * self.ch_cut_len: (num + 1) * self.ch_cut_len]
            gene_ar_cut = gene_ar[num * self.ch_cut_len: (num + 1) * self.ch_cut_len]

            gene_id = gene_ar_cut[mask_vec_cut,]
            assays = self.tracks[:, mask_vec_cut]
            temp_df = pd.DataFrame(assays.transpose(), columns=list(np.arange(cfg.input_size_encoder)))
            temp_df['gene_id'] = gene_id
            assay_matrix = assay_matrix.append(temp_df)

        return assay_matrix


if __name__ == '__main__':
    setup_logging()

    cfg = Config()
    data_ob_gene = DataPrepGene(cfg, mode='train')
    data_ob_gene.prepare_id_dict()
    data_gen = data_ob_gene.get_data()
    cut_seq = next(data_gen)
