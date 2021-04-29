from matplotlib import pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


class Viz:

    def __init__(self, cfg):
        self.cfg = cfg

    def plot_prediction(self, y_hat, y_true, mse, iter_num):
        start, end = (iter_num - 1) * self.cfg.cut_seq_len, iter_num * self.cfg.cut_seq_len
        x = np.arange(start, end)
        vocab = y_hat.shape[0]

        fig = plt.figure(1, figsize=(30, 20))
        gridspec.GridSpec(vocab + vocab - 1, 2)

        pos_l = np.arange(1, vocab + vocab / 2, 1)
        pos_l = [x for x in pos_l if x % 3 != 0]
        pos_l = pos_l - np.ones((1, len(pos_l)))
        pos = np.concatenate((pos_l, pos_l), axis=1)[0]
        pos = pos.astype(int)

        for i in range(vocab):
            plt.subplot2grid((vocab + vocab - 1, 2), (pos[2 * i], int(i / (vocab / 2))))
            plt.title("MSE : {0:.4f}".format(mse[0, i]), fontsize=6)
            plt.fill_between(x, 0, y_true[i, :], color='b')
            # plt.legend(fontsize=5)
            plt.ylim(-4, 4)
            plt.xlim(start, end)
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)

            plt.subplot2grid((vocab + vocab - 1, 2), (pos[2 * i + 1], int(i / (vocab / 2))))
            plt.fill_between(x, 0, y_hat[i, :], color='g')
            # plt.legend(fontsize=5)
            plt.ylim(-4, 4)
            plt.xlim(start, end)
            plt.xticks(fontsize=5)
            plt.yticks(fontsize=5)

        fig.savefig(self.cfg.result_path + '/' + 'track' + str(iter_num) + '.png')

        plt.close("all")
