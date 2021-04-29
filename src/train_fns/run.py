from train_fns import train_gene
from downstream.run_downstream import DownstreamTasks
from train_fns import config
import os
from train_fns.test_gene import get_config
import numpy as np
import logging
from common.log import setup_logging

logger = logging.getLogger(__name__)


def get_avg_map(mapdict_rna_seq):
    map_vec = []

    for k, v in mapdict_rna_seq.items():
        map_vec.append(v)

    mean_map = np.array(map_vec).mean()

    return mean_map


def run_all(chr_list, down_dir):
    config_base = 'config.yaml'
    result_base = 'down_images'

    for chr in chr_list:
        # cfg = config.Config()
        dir_name = down_dir + "/" + str(chr) + "/"
        model_dir_name = down_dir + "/model"

        """
        train_gene.train_iter_gene(cfg, chr=chr)

        os.system("mkdir {}".format(dir_name))
        os.system("mkdir {}".format(model_dir_name))
        os.system("mv -v {}/* {}/".format(cfg.model_dir, model_dir_name))
        """
        
        cfg = get_config(model_dir_name, config_base, result_base)
        pd_col = list(np.arange(cfg.hidden_size_encoder))
        pd_col.append('target')
        pd_col.append('gene_id')
        cfg = cfg._replace(downstream_df_columns=pd_col)

        downstream_ob = DownstreamTasks(cfg, dir_name, chr, mode='lstm')

        mapdict_rna_seq = downstream_ob.run_rna_seq(cfg)
        mean_map_rna = get_avg_map(mapdict_rna_seq)
        logging.info("mean MAP RNA-Seq: {}".format(mean_map_rna))

        mapdict_pe_seq = downstream_ob.run_pe(cfg)
        mean_map_pe = get_avg_map(mapdict_pe_seq)
        logging.info("mean MAP PE: {}".format(mean_map_pe))

        mapdict_fire_seq = downstream_ob.run_fires(cfg)
        mean_map_fire = get_avg_map(mapdict_fire_seq)
        logging.info("mean MAP fire: {}".format(mean_map_fire))

    return mapdict_rna_seq, mapdict_pe_seq, mapdict_pe_seq


if __name__ == "__main__":
    chr_list = [20, 21]
    down_dir = "..data/"

    logging.basicConfig(filename=down_dir + "/run_log_eval.txt",
                        level=logging.DEBUG)

    # hidden_nodes = [6, 12, 24, 36, 48, 60, 96, 110]

    mapdict_rna_seq, mapdict_pe_seq, mapdict_pe_seq = run_all(chr_list, down_dir)

    # np.save(down_dir + "/" + "map_norm.npy", map_list_norm)
