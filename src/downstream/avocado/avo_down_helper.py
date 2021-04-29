import logging
import numpy as np
import pandas as pd
import pickle
from sklearn.externals import joblib
from train_fns.test_gene import get_config
from keras.models import load_model
from downstream.avocado.run_avocado import AvocadoAnalysis
import yaml
import traceback
import json

logger = logging.getLogger(__name__)


class AvoDownstreamHelper:
    def __init__(self, cfg):
        self.chr_len = cfg.chr_len
        self.cfg = cfg
        self.cfg_down = None
        self.columns = cfg.downstream_df_columns

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

    def create_mask(self, window_labels):
        ind_list = []
        label_ar = np.zeros(self.chr_len)
        gene_ar = np.zeros(self.chr_len)
        mask_vec = np.ones(self.chr_len, bool)

        '''
        for i in range(window_labels.shape[0]):

            start = window_labels.loc[i, "start"]
            end = window_labels.loc[i, "end"]

            # print("gene : {} - start : {})".format(i, start))

            if start > self.chr_len or end > self.chr_len:
                break

            for j in range(end + 1 - start):
                ind_list.append(start - 1 + j)
                label_ar[start - 1 + j] = window_labels.loc[i, "target"]
                gene_ar[start - 1 + j] = i

        mask_vec = np.zeros(self.chr_len, bool)
        ind_ar = np.array(ind_list)

        mask_vec[ind_ar] = True
        '''

        return mask_vec, label_ar, gene_ar

    def get_feature_matrix(self, model_path, model_name, cfg, mask_vector, feature_matrix, run_features, label_ar,
                           gene_ar, feat_mat, mode):

        if run_features:
            Avocado_ob = AvocadoAnalysis()
            model = load_model("{}.h5".format(model_path + model_name))

            gen_factors = Avocado_ob.get_genomic_factors(model, cfg, mask_vector)
            feature_matrix = self.filter_states(gen_factors, feature_matrix,
                                                mask_vector, label_ar, gene_ar)

            # feature_matrix.to_pickle(feat_mat)
            joblib.dump(feature_matrix, feat_mat)

        else:

            feature_matrix = joblib.load(feat_mat)
            # feature_matrix = pd.read_pickle(feat_mat)
            feature_matrix.gene_id = feature_matrix.gene_id.astype(int)

            if not mode == 'pe':
                label_matrix = pd.DataFrame(columns=['target'])

                lab = label_ar[mask_vector,]
                lab = lab.reshape((-1, 1))

                label_matrix = label_matrix.append(pd.DataFrame(lab, columns=['target']),
                                                   ignore_index=True)

                label_matrix.target = label_matrix.target.astype(int)
                feature_matrix.target = label_matrix.target

        return feature_matrix

    def filter_states(self, gen_factors, feature_matrix, mask_vector, label_ar, gene_ar):

        # if True in mask_vector:
        #    print("here")
        # enc = avocado_features[mask_vector,]

        lab = label_ar[mask_vector,]
        lab = lab.reshape((gen_factors.shape[0], 1))
        gene_id = gene_ar[mask_vector,]
        gene_id = gene_id.reshape((gen_factors.shape[0], 1))

        feat_mat = np.append(gen_factors, lab, axis=1)
        feat_mat = np.append(feat_mat, gene_id, axis=1)

        feature_matrix = feature_matrix.append(pd.DataFrame(feat_mat, columns=self.columns),
                                               ignore_index=True)

        return feature_matrix


if __name__ == '__main__':
    file_name = ""
    config_base = 'config.yaml'
    result_base = 'down_images'
    model_path = ""

    cfg = get_config(model_path, config_base, result_base)
    helper_ob = AvoDownstreamHelper(cfg)

    feature_matrix = pd.read_pickle(file_name)
    feature_matrix.target = feature_matrix.target.astype(int)
