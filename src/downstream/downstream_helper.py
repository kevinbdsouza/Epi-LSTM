import logging
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import average_precision_score
from sklearn.utils import resample
from train_fns.test_gene import get_config
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class DownstreamHelper:
    def __init__(self, cfg, chr, mode):
        if mode == 'avocado':
            self.chr_len = cfg.chr_len
        elif mode == 'lstm':
            self.chr_len = cfg.chr_len[str(chr)]
        self.cfg = cfg
        self.cfg_down = None
        self.columns = cfg.downstream_df_columns

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

    def filter_states(self, encoder_hidden_states_np, feature_matrix, mask_vector, label_ar, gene_ar):

        # if True in mask_vector:
        #     print("here")

        enc = encoder_hidden_states_np[mask_vector,]
        lab = label_ar[mask_vector,]
        lab = lab.reshape((enc.shape[0], 1))
        gene_id = gene_ar[mask_vector,]
        gene_id = gene_id.reshape((enc.shape[0], 1))

        feat_mat = np.append(enc, lab, axis=1)
        feat_mat = np.append(feat_mat, gene_id, axis=1)

        if True in mask_vector:
            feature_matrix = feature_matrix.append(pd.DataFrame(feat_mat, columns=self.columns),
                                                   ignore_index=True)

        return feature_matrix

    def get_window_features(self, feature_matrix):

        col_length = self.cfg.input_size_encoder
        pd_col = list(np.arange(col_length))
        # pd_col.append('target')
        window_feature_matrix = pd.DataFrame(columns=pd_col)

        n_genes = feature_matrix['gene_id'].nunique()
        count = 0
        for i in range(n_genes):
            subset_gene_df = feature_matrix.loc[feature_matrix["gene_id"] == i,]

            if not subset_gene_df.empty:
                window_feature_matrix.loc[count, 0:col_length] = subset_gene_df.iloc[:,
                                                                                   0:col_length].mean()

                # window_feature_matrix.iloc[count]["target"] = subset_gene_df.iloc[0]["target"]
                count += 1

        window_feature_matrix = window_feature_matrix.apply(pd.to_numeric)
        # window_feature_matrix.target = window_feature_matrix.target.astype(int)

        return window_feature_matrix

    def plot_roc(self, y_test, y_hat):
        fpr, tpr, _ = roc_curve(y_test, y_hat)
        plt.plot(fpr, tpr, marker='o', markersize=14)
        plt.show()
        auc = roc_auc_score(y_test, y_hat)
        return auc

    def calculate_map3(self, feature_matrix, cls_mode):

        n_folds = 20
        hidden_size = None
        if cls_mode == 'concat':
            hidden_size = self.cfg_down.hidden_size_encoder
        elif cls_mode == 'ind':
            hidden_size = self.cfg.hidden_size_encoder

        average_precisions = np.zeros(n_folds)
        X_train = pd.DataFrame(columns=list(np.arange(hidden_size)))
        y_train = pd.DataFrame()
        feature_matrix = feature_matrix.sample(frac=1)

        y_test_cell = []
        y_hat_cell = []
        for i in range(n_folds):
            X_test = feature_matrix.iloc[i::n_folds, 0:hidden_size]
            X_valid = feature_matrix.iloc[(i + 1) % n_folds::n_folds, 0:hidden_size]
            y_test = feature_matrix.iloc[i::n_folds]["target"]
            y_valid = feature_matrix.iloc[(i + 1) % n_folds::n_folds]["target"]

            for j in range(n_folds):
                if j != i and j != (i + 1) % n_folds:
                    fold_mat = feature_matrix.iloc[j::n_folds, 0:hidden_size]
                    y_mat = feature_matrix.iloc[j::n_folds]["target"]
                    X_train = pd.concat([X_train, fold_mat])
                    y_train = pd.concat([y_train, y_mat])

            y_train = y_train.astype(int)

            model = xgboost.XGBClassifier(n_estimators=5000, nthread=min(X_train.shape[1], 12), max_depth=6)

            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='map', early_stopping_rounds=20,
                      verbose=False)

            y_hat = model.predict_proba(X_test)
            y_test_cell.append(y_test)
            y_hat_cell.append(y_hat)

        return y_test_cell, y_hat_cell

    def calculate_map2(self, feature_matrix, cls_mode):

        n_folds = 20
        hidden_size = None
        if cls_mode == 'concat':
            hidden_size = self.cfg_down.hidden_size_encoder
        elif cls_mode == 'ind':
            hidden_size = self.cfg.hidden_size_encoder

        average_precisions = np.zeros(n_folds)
        X_train = pd.DataFrame(columns=list(np.arange(hidden_size)))
        y_train = pd.DataFrame()
        feature_matrix = feature_matrix.sample(frac=1)

        for i in range(n_folds):
            X_test = feature_matrix.iloc[i::n_folds, 0:hidden_size]
            X_valid = feature_matrix.iloc[(i + 1) % n_folds::n_folds, 0:hidden_size]
            y_test = feature_matrix.iloc[i::n_folds]["target"]
            y_valid = feature_matrix.iloc[(i + 1) % n_folds::n_folds]["target"]

            for j in range(n_folds):
                if j != i and j != (i + 1) % n_folds:
                    fold_mat = feature_matrix.iloc[j::n_folds, 0:hidden_size]
                    y_mat = feature_matrix.iloc[j::n_folds]["target"]
                    X_train = pd.concat([X_train, fold_mat])
                    y_train = pd.concat([y_train, y_mat])

            y_train = y_train.astype(int)

            model = xgboost.XGBClassifier(n_estimators=5000, nthread=min(X_train.shape[1], 12), max_depth=6)

            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='map', early_stopping_rounds=20,
                      verbose=False)

            y_hat = model.predict_proba(X_test)
            average_precisions[i] = average_precision_score(y_test, y_hat[:, 1])

        mean_map = np.nanmean(average_precisions)

        if np.isnan(mean_map):
            mean_map = 0

        return mean_map

    def calculate_map(self, feature_matrix, cls_mode):

        n_folds = 1
        hidden_size = None
        if cls_mode == 'concat':
            hidden_size = self.cfg_down.hidden_size_encoder
        elif cls_mode == 'ind':
            hidden_size = self.cfg.hidden_size_encoder

        average_precisions = np.zeros(n_folds)
        for i in range(n_folds):
            msk_test = np.random.rand(len(feature_matrix)) < 0.8
            X_train_val = feature_matrix[msk_test].reset_index(drop=True)
            X_test = feature_matrix[~msk_test].reset_index(drop=True)

            msk_val = np.random.rand(len(X_train_val)) < 0.8
            X_train = X_train_val[msk_val].reset_index(drop=True)
            X_valid = X_train_val[~msk_val].reset_index(drop=True)

            model = xgboost.XGBClassifier(n_estimators=5000, nthread=min(X_train.shape[1] - 1, 12), max_depth=6)
            model.fit(X_train.iloc[:, 0:hidden_size], X_train[:]["target"],
                      eval_set=[(X_valid.iloc[:, 0:hidden_size], X_valid.iloc[:]["target"])],
                      eval_metric='map',
                      early_stopping_rounds=20,
                      verbose=False)

            y_hat = model.predict_proba(X_test.iloc[:, 0:hidden_size])

            average_precisions[i] = average_precision_score(X_test.iloc[:]["target"], y_hat[:, 1])

        mean_map = np.nanmean(average_precisions)

        return mean_map

    def get_feature_matrix(self, cfg, mask_vector, label_ar, gene_ar, run_features, feat_mat, downstream_main, chr):

        if run_features:
            feature_matrix = downstream_main(cfg, mask_vector, label_ar, gene_ar)
            feature_matrix.to_pickle(feat_mat)
            feature_matrix.target = feature_matrix.target.astype(int)
            feature_matrix.gene_id = feature_matrix.gene_id.astype(int)
        else:
            feature_matrix = pd.read_pickle(feat_mat)
            feature_matrix.gene_id = feature_matrix.gene_id.astype(int)

            label_matrix = pd.DataFrame(columns=['target'])

            for i in range(cfg.chr_len[str(chr)] // cfg.cut_seq_len):
                mask_vec_cut = mask_vector[i * cfg.cut_seq_len: (i + 1) * cfg.cut_seq_len]
                label_ar_cut = label_ar[i * cfg.cut_seq_len: (i + 1) * cfg.cut_seq_len]

                lab = label_ar_cut[mask_vec_cut,]
                lab = lab.reshape((-1, 1))

                label_matrix = label_matrix.append(pd.DataFrame(lab, columns=['target']),
                                                   ignore_index=True)

            label_matrix.target = label_matrix.target.astype(int)
            feature_matrix.target = label_matrix.target

        return feature_matrix

    def balance_classes(self, feature_matrix):
        if feature_matrix["target"].value_counts().index[0] == 1:
            bal_mode = "undersampling"
        else:
            bal_mode = "oversampling"

        feature_matrix = self.fix_class_imbalance(feature_matrix, mode=bal_mode)

        return feature_matrix

    def fix_class_imbalance(self, feature_matrix, mode='undersampling'):

        if mode == 'undersampling':
            feat_majority = feature_matrix[feature_matrix.target == 1]
            feat_minority = feature_matrix[feature_matrix.target == 0]

            feat_majority_downsampled = resample(feat_majority,
                                                 replace=False,
                                                 n_samples=feat_minority.shape[0],
                                                 random_state=123)

            feature_matrix = pd.concat([feat_majority_downsampled, feat_minority]).reset_index(drop=True)

        elif mode == 'oversampling':
            feat_majority = feature_matrix[feature_matrix.target == 0]
            feat_minority = feature_matrix[feature_matrix.target == 1]

            feat_minority_upsampled = resample(feat_minority,
                                               replace=True,
                                               n_samples=feat_majority.shape[0],
                                               random_state=123)

            feature_matrix = pd.concat([feat_minority_upsampled, feat_majority]).reset_index(drop=True)

        return feature_matrix

    def concat_gene_features(self, feature_matrix):

        n_genes = feature_matrix['gene_id'].nunique()
        max_gene_length = np.max(feature_matrix['gene_id'].value_counts())

        new_feature_mat = np.zeros((n_genes, max_gene_length * self.cfg.hidden_size_encoder))

        for i in range(n_genes):

            subset_gene_df = feature_matrix.loc[feature_matrix["gene_id"] == i,]
            concat_feature = []

            # print("gene : {} - df shape : {})".format(i, subset_gene_df.shape[0]))

            for j in range(subset_gene_df.shape[0]):
                concat_feature = np.append(concat_feature, subset_gene_df.iloc[j, 0:self.cfg.hidden_size_encoder])

            new_feature_mat[i, 0:len(concat_feature)] = np.array(concat_feature)

        return new_feature_mat


if __name__ == '__main__':
    file_name = ""
    config_base = 'config.yaml'
    result_base = 'down_images'
    model_path = ""

    cfg = get_config(model_path, config_base, result_base)
    helper_ob = DownstreamHelper(cfg)

    feature_matrix = pd.read_pickle(file_name)
    feature_matrix.target = feature_matrix.target.astype(int)

    balanced_feat = helper_ob.fix_class_imbalance(feature_matrix, mode='oversampling')
