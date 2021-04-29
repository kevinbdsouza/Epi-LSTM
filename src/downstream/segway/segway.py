import logging
from Bio import SeqIO
import pandas as pd
import numpy as np
# from segway import run
from plastid import GenomicSegment, Transcript
import os
import pickle
from scipy.spatial import distance

logger = logging.getLogger(__name__)


class SegWay:
    def __init__(self):
        self.fasta_path = "/data2/latent/data/dna/chr21.fa"
        self.dir = "../lstm_features/"
        self.features_path = self.dir + "feat_chr_21.pkl"

    def read_fasta(self, fasta_path):
        records = list(SeqIO.parse(fasta_path, "fasta"))
        new_seq = records[0].seq

        return new_seq

    def convert_to_bp_resolution(self, track):
        bp_track = np.zeros((25 * len(track, )))

        for i in range(len(track)):
            bp_track[25 * i:(i + 1) * 25 - 1] = track[i]

        return bp_track

    def create_signal_from_pickle(self, feature_path):
        signal = None

        for i in range(0, 24):
            features = pd.read_pickle(self.features_path)
            feature = features.loc[:, i]

            # bp_track = self.convert_to_bp_resolution(np.array(feature))

            for j in range(len(np.array(feature))):
                with open(self.dir + 'feature_' + str(i) + '.txt', "a") as myfile:
                    line = 'chr21   ' + str(25 * j + 1) + "   " + str(25 * (j + 1)) + "    " + str(feature[j]) + '\n'

                    myfile.write(line)

        return signal

    def load_features(self, chr_len):

        for i in range(11, 24):
            feature_path = self.dir + 'feature_' + str(i) + '.npy'
            feature = np.load(feature_path)
            feature = feature[:chr_len]

            with open(self.dir + 'feature_' + str(i) + '.wigFix', "ab") as myfile:
                np.savetxt(myfile, feature, delimiter=',', newline='\n')

        return

    def rename_files(self):
        for filename in os.listdir(self.dir):
            os.rename(filename, filename)

        pass

    def run_genome_euclid(self):
        target = open(
            '../lstm_features/feat_chr_21.pkl',
            'rb')
        chr_df = pickle.load(target)
        target.close()

        chr_df = chr_df.drop(['target', 'gene_id'], axis=1)

        all_means = []
        for i in range(1, len(chr_df) - 1):
            all_dst = []
            diff = i
            for k in range(0, len(chr_df) - 1):
                if k + diff >= len(chr_df):
                    break
                a = chr_df.iloc[k, :]
                b = chr_df.iloc[k + diff, :]
                all_dst.append(distance.euclidean(a, b))
            all_means.append(np.mean(all_dst))

        return all_means

    def run_segway(self):

        GENOMEDATA_DIRNAME = "../lstm_features/genomedata/genomedata.test"

        # run.main(["--random-starts=3", "train", GENOMEDATA_DIRNAME])

        return


if __name__ == '__main__':
    seg_ob = SegWay()
    fasta_seq = seg_ob.read_fasta(seg_ob.fasta_path)

    chr_len = len(fasta_seq)

    # signal = seg_ob.create_signal_from_pickle(seg_ob.features_path)
    # seg_ob.load_features(chr_len)

    # seg_ob.run_segway()
    all_means = seg_ob.run_genome_euclid()
    print("done")
