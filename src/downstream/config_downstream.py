import os


class ConfigDownstream:
    def __init__(self):
        self.network = 'lstm'
        self.load_weights = False

        self.input_size_encoder = 1
        self.hidden_size_encoder = 24

        self.input_size_decoder = self.hidden_size_encoder
        self.hidden_size_decoder = self.hidden_size_encoder
        self.output_size_decoder = self.input_size_encoder

        self.learning_rate = 1e-2

        self.cut_seq_len = 100

        self.model_dir = ''
        self.config_base = 'config_down.yaml'
        self.tensorboard_log_base = 't_log'
        self.config_file = os.path.join(self.model_dir, self.config_base)
        self.tensorboard_log_path = os.path.join(self.model_dir, self.tensorboard_log_base)

        if not os.path.exists(self.tensorboard_log_path):
            os.makedirs(self.tensorboard_log_path)

        self.num_epochs = 2
        self.chr21_len = 1926400
