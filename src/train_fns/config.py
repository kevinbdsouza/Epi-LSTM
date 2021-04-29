import os


class Config:
    def __init__(self):
        self.network = 'lstm'
        self.load_weights = False

        self.input_size_encoder = 1030
        self.hidden_size_encoder = 3

        self.cell_assay_embed_size = self.hidden_size_encoder

        self.input_size_decoder = self.hidden_size_encoder
        self.hidden_size_decoder = self.hidden_size_encoder
        self.output_size_decoder = self.input_size_encoder

        self.learning_rate = 1e-2
        self.max_norm = 5e-14

        self.cut_seq_len = 100
        self.base_pair_resolution = 25
        self.use_dna_seq = False

        self.fasta_path = "/opt/data/latent/data/dna"

        self.epigenome_npz_path_train = '/data2/latent/data/npz/chr21_arc_sinh_znorm'
        self.epigenome_npz_path_test = '/data2/latent/data/npz/chr21_arc_sinh_znorm'
        self.epigenome_bigwig_path = '/opt/data/latent/data/bigwig'

        self.model_dir = '..data/'
        self.config_base = 'config.yaml'
        self.tensorboard_log_base = 't_log'
        self.config_file = os.path.join(self.model_dir, self.config_base)
        self.tensorboard_log_path = os.path.join(self.model_dir, self.tensorboard_log_base)

        if not os.path.exists(self.tensorboard_log_path):
            os.makedirs(self.tensorboard_log_path)

        self.data_dir = '.data/'
        self.num_epochs = 3

        self.chr_cuts = {'1': 25, '2': 25, '3': 20, '4': 20, '5': 19, '6': 18,
                         '7': 17, '8': 16, '9': 15, '10': 15, '11': 14, '12': 14,
                         '13': 12, '14': 11, '15': 11, '16': 10, '17': 9, '18': 8,
                         '19': 7, '20': 7, '21': 5}
        self.chr_len = {'1': 9900000, '2': 9800000, '3': 8000000, '4': 7700000, '5': 7300000, '6': 6840000,
                        '7': 6400000, '8': 5900000, '9': 5700000, '10': 5500000, '11': 5400000, '12': 5300000,
                        '13': 4570000, '14': 4260000, '15': 4020000, '16': 3560000, '17': 3160000, '18': 3050000,
                        '19': 2560000, '20': 2518700,
                        '21': 1926400}

        # self.downstream_df_columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13',
        #                              'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24',
        #                              'target', 'gene_id']

        self.downstream_df_columns = ['f1', 'f2', 'f3', 'target', 'gene_id']
