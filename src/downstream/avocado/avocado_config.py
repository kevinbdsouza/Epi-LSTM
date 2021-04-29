import os


class Config:
    def __init__(self):
        self.network = 'lstm'
        self.load_weights = False

        self.num_layers = 2
        self.num_nodes = 2048
        self.assay_factors = 256
        self.ca_factors = 32
        self.bp25_factors = 25
        self.bp250_factors = 40
        self.bp5k_factors = 45
        self.batch_size = 10000
        self.learning_rate = 1e-2
        self.epoch_size = 193
        self.num_epochs = 10
        self.base_pair_resolution = 25
        self.hidden_size_encoder = self.bp250_factors + self.bp25_factors + self.bp5k_factors

        self.fasta_path = "/opt/data/latent/data/dna"
        self.epigenome_npz_path_train = '/opt/data/latent/data/npz/all_npz_arc_sinh'
        self.epigenome_npz_path_test = '/opt/data/latent/data/npz/all_npz_arc_sinh'
        self.epigenome_bigwig_path = '/opt/data/latent/data/bigwig'

        self.model_dir = ''
        self.config_base = 'avocado_config.yaml'
        self.tensorboard_log_base = 't_log'
        self.config_file = os.path.join(self.model_dir, self.config_base)
        self.tensorboard_log_path = os.path.join(self.model_dir, self.tensorboard_log_base)

        if not os.path.exists(self.tensorboard_log_path):
            os.makedirs(self.tensorboard_log_path)

        self.data_dir = '.data/'
        self.chr21_len = 1925195
        self.downstream_df_columns = None

