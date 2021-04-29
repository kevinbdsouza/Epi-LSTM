import numpy as np
from os import listdir
from downstream.avocado.model import Avocado
from downstream.avocado import avocado_config as config
from downstream.avocado import avo_down_helper as helper
import logging
from os.path import isfile, join
from keras.models import load_model
from train_fns.monitor_training import MonitorTraining
import re

gpu_id = 1
mode = 'train'

logger = logging.getLogger(__name__)


class AvocadoAnalysis:
    def __init__(self):
        self.epigenome_list = []
        self.assay_list = []
        self.data_avocado = {}
        self.epigenome_npz_path = None
        self.model_name = "avocado-chr21"
        self.vocab_size = None
        self.model_path = ""

    def get_data(self, cfg, mode):

        if mode == 'train':
            self.epigenome_npz_path = cfg.epigenome_npz_path_train
        elif mode == 'test':
            self.epigenome_npz_path = cfg.epigenome_npz_path_train

        npz_files = [f for f in listdir(self.epigenome_npz_path) if isfile(join(self.epigenome_npz_path, f))]
        npz_files.sort()

        for id, file in enumerate(npz_files[:100]):
            epigenome_assay = re.split(r"\-\s*", re.split(r"\.\s*", file)[0])

            self.data_avocado[(epigenome_assay[0], epigenome_assay[1])] = \
                np.load(self.epigenome_npz_path + '/' + file)['arr_0'][0]

            if epigenome_assay[0] not in self.epigenome_list:
                self.epigenome_list.append(epigenome_assay[0])

            if epigenome_assay[1] not in self.assay_list:
                self.assay_list.append(epigenome_assay[1])

        self.vocab_size = len(self.data_avocado)

    def train_avocado(self, cfg):

        model = Avocado(self.epigenome_list, self.assay_list, n_layers=cfg.num_layers,
                        n_genomic_positions=cfg.chr21_len,
                        n_nodes=cfg.num_nodes, n_assay_factors=cfg.assay_factors,
                        n_celltype_factors=cfg.ca_factors,
                        n_25bp_factors=cfg.bp25_factors, n_250bp_factors=cfg.bp250_factors,
                        n_5kbp_factors=cfg.bp5k_factors, batch_size=cfg.batch_size)

        model.summary()

        model.fit(self.data_avocado, n_epochs=cfg.num_epochs, epoch_size=cfg.epoch_size)

        model.save("{}.h5".format(cfg.model_dir + self.model_name))

        return model

    def test_avocado(self, model, celltype, assay):

        y_pred = model.predict(celltype, assay)

        return y_pred

    def get_genomic_factors(self, model, cfg, mask_vector):

        n_25bp = cfg.bp25_factors
        n_250bp = cfg.bp250_factors
        n_5kbp = cfg.bp5k_factors

        mask_len = np.count_nonzero(mask_vector)
        gen_factors = np.empty((mask_len,
                                n_25bp + n_250bp + n_5kbp))

        for layer in model.layers:
            if layer.name == 'genome_25bp_embedding':
                genome_25bp_embedding = layer.get_weights()[0]
            elif layer.name == 'genome_250bp_embedding':
                genome_250bp_embedding = layer.get_weights()[0]
            elif layer.name == 'genome_5kbp_embedding':
                genome_5kbp_embedding = layer.get_weights()[0]

        n1 = n_25bp
        n2 = n_25bp + n_250bp

        pos = 0
        for i in range(cfg.chr_len):

            if mask_vector[i]:
                gen_factors[pos, :n1] = genome_25bp_embedding[i]
                gen_factors[pos, n1:n2] = genome_250bp_embedding[i // 10]
                gen_factors[pos, n2:] = genome_5kbp_embedding[i // 200]

                pos += 1
                
        return gen_factors

    def run_avocado(self, cfg):

        self.get_data(cfg, mode='train')

        model = self.train_avocado(cfg)

        monitor = MonitorTraining(cfg, self.vocab_size)
        monitor.save_config_as_yaml(cfg.config_file, cfg)

        return model


if __name__ == "__main__":
    cfg = config.Config()

    AvDown_ob = helper.AvoDownstreamHelper(cfg)
    Avocado_ob = AvocadoAnalysis()

    # pd_col = list(np.arange(cfg.bp25_factors + cfg.bp250_factors + cfg.bp5k_factors))
    # pd_col.append('target')
    # cfg.downstream_df_columns = pd_col

    AvDown_ob.save_config_as_yaml(cfg.config_file, cfg)
    # model = Avocado_ob.run_avocado(cfg)

    # model = load_model("{}.h5".format(Avocado_ob.model_path + Avocado_ob.model_name))

    # gen_factors = Avocado_ob.get_genomic_factors(model, cfg)

    print("done")
