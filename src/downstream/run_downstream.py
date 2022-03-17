import logging
from downstream.rna_seq import RnaSeq
from downstream.pe_interactions import PeInteractions
from train_fns.data_prep_gene import DataPrepGene
from train_fns.monitor_testing import MonitorTesting
from train_fns.train_gene import unroll_loop
from train_fns.test_gene import get_config
from common.log import setup_logging
from keras.callbacks import TensorBoard
from eda.viz import Viz
from train_fns.model import Model
import traceback
import numpy as np
import pandas as pd
from downstream.downstream_helper import DownstreamHelper
from downstream.downstream_lstm import DownstreamLSTM
from downstream.fires import Fires
from downstream.rep_timing import Rep_timing

gpu_id = 1
mode = "test"

logger = logging.getLogger(__name__)


class DownstreamTasks:
    def __init__(self, cfg, dir, chr, mode):
        self.data_dir = "/data2/latent/data/"
        self.rna_seq_path = self.data_dir + "downstream/RNA-seq"
        self.pe_int_path = self.data_dir + "downstream/PE-interactions"
        self.fire_path = self.data_dir + "downstream/FIREs"
        self.rep_timing_path = self.data_dir + "downstream/replication_timing"
        self.fire_cell_names = ['GM12878', 'H1', 'IMR90', 'MES', 'MSC', 'NPC', 'TRO']
        self.rep_cell_names = ['HUVEC', 'IMR90', 'K562']
        self.chr = chr
        self.pe_cell_names = ['E123', 'E117', 'E116', 'E017']
        self.chr_list_rna = str(chr)
        self.chr_list_pe = 'chr' + str(chr)
        self.chr_list_tad = 'chr' + str(chr)
        self.chr_list_rep = 'chr' + str(chr)
        self.chr_list_fire = chr
        self.saved_model_dir = dir
        self.feat_mat_rna = self.saved_model_dir + "feat_chr_" + str(chr) + "_rna"
        self.feat_mat_pe = self.saved_model_dir + "feat_chr_" + str(chr) + "_pe_"
        self.feat_mat_fire = self.saved_model_dir + "feat_chr_" + str(chr) + "_fire"
        self.feat_mat_rep = self.saved_model_dir + "feat_chr_" + str(chr) + "_rep"
        self.new_features = self.saved_model_dir + "new_feat_.npy"
        self.assay_path = self.saved_model_dir + "assay/new_positions/" + "assay_" + str(chr)
        self.run_features_rna = True
        self.run_features_pe = False
        self.run_features_fire = False
        self.run_features_rep = False
        self.concat_lstm = False
        self.run_concat_feat = False
        self.calculate_map = True
        self.roc = True
        self.downstream_helper_ob = DownstreamHelper(cfg, chr, mode=mode)
        self.down_lstm_ob = DownstreamLSTM()

    def downstream_main(self, cfg, mask_vector, label_ar, gene_ar):

        cfg = cfg._replace(epigenome_npz_path_train=self.data_dir + "npz/chr" + str(self.chr) + "_arc_sinh_znorm")
        cfg = cfg._replace(epigenome_npz_path_test=self.data_dir + "npz/chr" + str(self.chr) + "_arc_sinh_znorm")

        data_ob_gene = DataPrepGene(cfg, mode='test', chr=str(self.chr))
        monitor = MonitorTesting(cfg)
        callback = TensorBoard(cfg.tensorboard_log_path)

        data_ob_gene.prepare_id_dict()
        data_gen_test = data_ob_gene.get_data()
        model = Model(cfg, data_ob_gene.vocab_size, gpu_id)
        model.load_weights()
        model.set_callback(callback)

        logging.info('Downstream Start')

        iter_num = 0
        hidden_states = np.zeros((2, cfg.hidden_size_encoder))

        encoder_init = True
        decoder_init = True
        encoder_optimizer, decoder_optimizer, criterion = None, None, None

        feature_matrix = pd.DataFrame(columns=cfg.downstream_df_columns)

        try:
            for track_cut in data_gen_test:

                mse, hidden_states, encoder_init, decoder_init, predicted_cut, encoder_hidden_states_np = unroll_loop(
                    cfg, track_cut, model,
                    encoder_optimizer,
                    decoder_optimizer,
                    criterion,
                    hidden_states, encoder_init,
                    decoder_init, mode)

                mask_vector_cut = mask_vector[iter_num * cfg.cut_seq_len: (iter_num + 1) * cfg.cut_seq_len]
                label_ar_cut = label_ar[iter_num * cfg.cut_seq_len: (iter_num + 1) * cfg.cut_seq_len]
                gene_ar_cut = gene_ar[iter_num * cfg.cut_seq_len: (iter_num + 1) * cfg.cut_seq_len]

                feature_matrix = self.downstream_helper_ob.filter_states(encoder_hidden_states_np, feature_matrix,
                                                                         mask_vector_cut,
                                                                         label_ar_cut, gene_ar_cut)

                iter_num += 1

                if iter_num % 500 == 0:
                    logger.info('Iter: {} - mse: {}'.format(iter_num, np.mean(monitor.mse_iter)))
                    # vizOb.plot_prediction(predicted_cut, track_cut, mse, iter_num)

                monitor.monitor_mse_iter(callback, np.sum(mse), iter_num)

        except Exception as e:
            logger.error(traceback.format_exc())

        # print('Mean MSE at end of Downstream Run: {}'.format(np.mean(monitor.mse_iter)))

        return feature_matrix

    def run_rna_seq(self, cfg):
        logging.info("RNA-Seq start")

        rna_seq_ob = RnaSeq()
        rna_seq_ob.get_rna_seq(self.rna_seq_path)
        rna_seq_chr = rna_seq_ob.filter_rna_seq(self.chr_list_rna)

        rna_seq_chr['target'] = 0
        mean_map_dict = {}

        y_test = []
        y_hat = []
        for col in range(1, 58):
            rna_seq_chr.loc[rna_seq_chr.iloc[:, col] >= 0.5, 'target'] = 1
            rna_window_labels = rna_seq_chr.filter(['start', 'end', 'target'], axis=1)
            rna_window_labels = rna_window_labels.drop_duplicates(keep='first').reset_index(drop=True)
            rna_window_labels = rna_window_labels.drop([410, 598]).reset_index(drop=True)

            mask_vector, label_ar, gene_ar = self.downstream_helper_ob.create_mask(rna_window_labels)

            feature_matrix = self.downstream_helper_ob.get_feature_matrix(cfg, mask_vector, label_ar, gene_ar,
                                                                          self.run_features_rna,
                                                                          self.feat_mat_rna + '.pkl',
                                                                          self.downstream_main, self.chr)

            feature_matrix = self.downstream_helper_ob.get_window_features(feature_matrix)

            self.run_features_rna = False

            logging.info("chr : {} - cell : {}".format(str(self.chr), rna_seq_chr.columns[col]))

            if self.concat_lstm:

                if self.run_concat_feat:
                    new_feature_mat = self.downstream_helper_ob.concat_gene_features(feature_matrix)
                    new_feature_mat.to_pickle(self.new_features)
                else:
                    new_feature_mat = np.load(self.new_features)

                target = np.array(rna_window_labels[:]["target"])
                feature_matrix, cfg_down = self.down_lstm_ob.get_features(new_feature_mat, target)
                self.downstream_helper_ob.cfg.down = cfg_down
                cls_mode = 'concat'
            else:
                feature_matrix = feature_matrix.loc[:, feature_matrix.columns != 'gene_id']
                cls_mode = 'ind'

            if self.calculate_map:
                mean_map = self.downstream_helper_ob.calculate_map2(feature_matrix, cls_mode)
                mean_map_dict[rna_seq_chr.columns[col]] = mean_map

            if self.roc:
                y_test_cell, y_hat_cell = self.downstream_helper_ob.calculate_map3(feature_matrix, cls_mode)
                y_test.append(y_test_cell)
                y_hat.append(y_hat_cell)

        if self.roc:
            auc = self.downstream_helper_ob.plot_roc(y_test, y_hat)
        elif self.calculate_map:
            np.save(self.saved_model_dir + 'map_dict_rnaseq.npy', mean_map_dict)

        print("done")
        return mean_map_dict

    def run_pe(self, cfg):
        logging.info("PE start")

        pe_ob = PeInteractions()
        pe_ob.get_pe_data(self.pe_int_path)
        pe_data_chr = pe_ob.filter_pe_data(self.chr_list_pe)
        mean_map_dict = {}
        cls_mode = 'ind'

        for cell in self.pe_cell_names:
            pe_data_chr_cell = pe_data_chr.loc[pe_data_chr['cell'] == cell]
            pe_window_labels = pe_data_chr_cell.filter(['window_start', 'window_end', 'label'], axis=1)
            pe_window_labels.rename(columns={'window_start': 'start', 'window_end': 'end', 'label': 'target'},
                                    inplace=True)
            pe_window_labels = pe_window_labels.drop_duplicates(keep='first').reset_index(drop=True)

            mask_vector, label_ar, gene_ar = self.downstream_helper_ob.create_mask(pe_window_labels)

            feature_matrix = self.downstream_helper_ob.get_feature_matrix(cfg, mask_vector, label_ar, gene_ar,
                                                                          self.run_features_pe,
                                                                          self.feat_mat_pe + cell + ".pkl",
                                                                          self.downstream_main, self.chr)

            feature_matrix = self.downstream_helper_ob.get_window_features(feature_matrix)

            logging.info("chr : {} - cell : {}".format(str(self.chr), cell))

            if self.calculate_map:
                mean_map = self.downstream_helper_ob.calculate_map2(feature_matrix, cls_mode)

                mean_map_dict[cell] = mean_map

        np.save(self.saved_model_dir + 'map_dict_pe.npy', mean_map_dict)

        return mean_map_dict

    def run_fires(self, cfg):
        logging.info("fires start")

        fire_ob = Fires()
        fire_ob.get_fire_data(self.fire_path)
        fire_labeled = fire_ob.filter_fire_data(self.chr_list_fire)
        mean_map_dict = {}
        cls_mode = 'ind'

        for cell in self.fire_cell_names:
            fire_window_labels = fire_labeled.filter(['start', 'end', cell + '_l'], axis=1)
            fire_window_labels.rename(columns={cell + '_l': 'target'}, inplace=True)
            fire_window_labels = fire_window_labels.drop_duplicates(keep='first').reset_index(drop=True)

            mask_vector, label_ar, gene_ar = self.downstream_helper_ob.create_mask(fire_window_labels)

            feature_matrix = self.downstream_helper_ob.get_feature_matrix(cfg, mask_vector, label_ar, gene_ar,
                                                                          self.run_features_fire,
                                                                          self.feat_mat_fire + ".pkl",
                                                                          self.downstream_main, self.chr)

            feature_matrix = self.downstream_helper_ob.get_window_features(feature_matrix)

            self.run_features_fire = False

            logging.info("chr : {} - cell : {}".format(str(self.chr), cell))

            if self.calculate_map:
                mean_map = self.downstream_helper_ob.calculate_map2(feature_matrix, cls_mode)

                mean_map_dict[cell] = mean_map

        np.save(self.saved_model_dir + 'map_dict_fire.npy', mean_map_dict)

        return mean_map_dict

    def run_rep_timings(self, cfg):
        logging.info("rep start")

        rep_ob = Rep_timing()
        rep_ob.get_rep_data(self.rep_timing_path, self.rep_cell_names)
        rep_filtered = rep_ob.filter_rep_data(self.chr_list_rep)
        mean_map_dict = {}
        cls_mode = 'ind'

        for i, cell in enumerate(self.rep_cell_names):

            rep_data_cell = rep_filtered[i]
            rep_data_cell = rep_data_cell.filter(['start', 'end', 'target'], axis=1)
            rep_data_cell = rep_data_cell.drop_duplicates(keep='first').reset_index(drop=True)

            mask_vector, label_ar, gene_ar = self.downstream_helper_ob.create_mask(rep_data_cell)

            feature_matrix = self.downstream_helper_ob.get_feature_matrix(cfg, mask_vector, label_ar, gene_ar,
                                                                          self.run_features_rep,
                                                                          self.feat_mat_rep + ".pkl",
                                                                          self.downstream_main, self.chr)

            feature_matrix = self.downstream_helper_ob.get_window_features(feature_matrix)

            self.run_features_rep = False

            logging.info("chr : {} - cell : {}".format(str(self.chr), cell))

            if self.calculate_map:
                mean_map = self.downstream_helper_ob.calculate_map2(feature_matrix, cls_mode)

                mean_map_dict[cell] = mean_map

        np.save(self.saved_model_dir + 'map_dict_fire.npy', mean_map_dict)

        return mean_map_dict

    def get_assays_at_positions(self, cfg):

        data_prep_ob = DataPrepGene(cfg, mode='test', chr=str(self.chr))
        data_prep_ob.prepare_id_dict()
        element_list = ['promoter', 'enhancer']

        # enahncers
        '''
        e_path = "/data2/latent/data/interpretation/enhancers/enhancers.txt"
        enhancers = pd.read_table(e_path, delim_whitespace=True)
        enhancer_positions = enhancers['Id']
        enhancer_positions = enhancer_positions.loc[37528:38586].reset_index(drop=True)
        enhancer_positions = enhancer_positions.str.replace("chr21:", "")

        enhancer_positions = enhancer_positions.str.split("-", n=1, expand=True)
        enhancer_positions = enhancer_positions.astype(int)//25
        enhancer_positions.columns = ['start', 'end']
        enhancer_positions["target"] = 0

        mask_vector, label_ar, gene_ar = self.downstream_helper_ob.create_mask(enhancer_positions)

        assay_matrix = data_prep_ob.get_assay_data_at_positions(cfg, mask_vector, gene_ar)

        assay_matrix = self.downstream_helper_ob.get_window_features(assay_matrix)

        assay_matrix.to_pickle(self.assay_path + "_" + element_list[1])
        '''


        # promoters
        rna_seq_ob = RnaSeq()
        rna_seq_ob.get_rna_seq(self.rna_seq_path)
        rna_seq_chr = rna_seq_ob.filter_rna_seq(self.chr_list_rna)
        promoter_positions = rna_seq_chr['start']

        mask_vector, label_ar, gene_ar = self.downstream_helper_ob.create_mask(promoter_positions)
        assay_matrix = data_prep_ob.get_assay_data_at_positions(cfg, mask_vector, gene_ar)
        assay_matrix.to_pickle(self.assay_path + "_" + element_list[0])

        print("done")

        '''
        pe_ob = PeInteractions()
        data_prep_ob = DataPrepGene(cfg, mode='test', chr=str(self.chr))
        data_prep_ob.prepare_id_dict()
        pe_ob.get_pe_data(self.pe_int_path)
        pe_data_chr = pe_ob.filter_pe_data(self.chr_list_pe)
        element_list = ['promoter', 'enhancer']
        
        for e in element_list:
            for cell in self.pe_cell_names:
                pe_data_chr_cell = pe_data_chr.loc[pe_data_chr['cell'] == cell]
                pe_window_labels = pe_data_chr_cell.filter([e + '_start', e + '_end', 'label'], axis=1)
                pe_window_labels.rename(columns={e + '_start': 'start', e + '_end': 'end', 'label': 'target'},
                                        inplace=True)
                pe_window_labels = pe_window_labels.drop_duplicates(keep='first').reset_index(drop=True)

                mask_vector, label_ar, gene_ar = self.downstream_helper_ob.create_mask(pe_window_labels)

                assay_matrix = data_prep_ob.get_assay_data_at_positions(cfg, mask_vector, gene_ar)

                assay_matrix = self.downstream_helper_ob.get_window_features(assay_matrix)

                assay_matrix.to_pickle(self.assay_path + "_" + e + '_' + cell)

        '''

        return assay_matrix


if __name__ == '__main__':
    setup_logging()
    config_base = 'config.yaml'
    result_base = 'down_images'
    chr = 21

    dir = "../data"
    model_path = "../data"
    cfg = get_config(model_path, config_base, result_base)
    pd_col = list(np.arange(cfg.hidden_size_encoder))
    pd_col.append('target')
    pd_col.append('gene_id')
    cfg = cfg._replace(downstream_df_columns=pd_col)

    downstream_ob = DownstreamTasks(cfg, dir, chr, mode='lstm')

    mapdict_rna_seq = downstream_ob.run_rna_seq(cfg)

    # mapdict_pe = downstream_ob.run_pe(cfg)

    # mapdict_fire = downstream_ob.run_fires(cfg)

    # mapdict_rep = downstream_ob.run_rep_timings(cfg)

    # mapdict_fire = downstream_ob.run_fires(cfg)

    # assay_matrix = downstream_ob.get_assays_at_positions(cfg)

    # mapdict_fire = downstream_ob.run_fires(cfg)

    # mapdict_fire = downstream_ob.run_fires(cfg)

    # mapdict_fire = downstream_ob.run_fires(cfg)

    print("done")
