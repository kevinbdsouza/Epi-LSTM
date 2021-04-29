import numpy
import sys

from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score
from avocado import *

data_dir = '/net/noble/vol4/noble/user/jmschr/proj/avocado/data'
peak_dir = '/net/noble/vol5/user/jmschr/proj/avocado/peaks'
pred_dir = '/net/noble/vol5/user/jmschr/proj/avocado/predictions'

def mse(y_true, y_pred):
	return ((y_true - y_pred) ** 2.).mean()

def extract_indices(celltype, assay, chrom, idxs):
	try:
		y_true = numpy.load('{}/{}.{}.chr{}.arcsinh.npy'.format(data_dir, celltype, assay, chrom))[idxs].astype('float32')
		y_chrom = numpy.load('{}/ChromImpute/{}.{}.chr{}.imputed.npz'.format(pred_dir, celltype, assay, chrom))['arr_0'][idxs].astype('float32')
		y_predictd = numpy.load('{}/PREDICTD/{}.{}.chr{}.predictd.npz'.format(pred_dir, celltype, assay, chrom))['arr_0'][idxs].astype('float32')
		y_avo = numpy.load('{}/avocado/{}.{}.chr{}.avocado.npz'.format(pred_dir, celltype, assay, chrom))['arr_0'][idxs].astype('float32')
		return y_true, y_chrom, y_predictd, y_avo
	except:
		return [], [], [], []

assay, chrom, count = sys.argv[1:]
chrom, count = int(chrom), int(count)

n_peaks = numpy.zeros(chromosome_lengths[chrom-1])
peaks = []

for celltype in celltypes:
	try:
		peak = numpy.load('{}/{}.{}.chr{}.narrowPeak.npz'.format(peak_dir, celltype, assay, chrom))['arr_0'].astype('int8')
		n_peaks += peak
		peaks.append(peak)
	except:
		continue

if n_peaks.max() < count:
	sys.exit()

idxs = (n_peaks > 0) & (n_peaks <= count)
idxs = n_peaks == count

y_peaks    = numpy.concatenate([peak[idxs] for peak in peaks])
data       = Parallel(n_jobs=1)(delayed(extract_indices)(celltype, assay, chrom, idxs) for celltype in celltypes)
y_true     = numpy.concatenate([datum[0] for datum in data])
y_chrom    = numpy.concatenate([datum[1] for datum in data])
y_predictd = numpy.concatenate([datum[2] for datum in data])
y_avo      = numpy.concatenate([datum[3] for datum in data])

scores = numpy.zeros((4, 5))
t = numpy.arcsinh(2)
scores[0] = roc_auc_score(y_peaks, y_true), average_precision_score(y_peaks, y_true), -1, recall_score(y_peaks, (y_true > t).astype(int)), precision_score(y_peaks, (y_true > t).astype(int))
scores[1] = roc_auc_score(y_peaks, y_chrom), average_precision_score(y_peaks, y_chrom), mse(y_true, y_chrom), recall_score(y_peaks, (y_chrom > t).astype(int)), precision_score(y_peaks, (y_chrom > t).astype(int))
scores[2] = roc_auc_score(y_peaks, y_predictd), average_precision_score(y_peaks, y_predictd), mse(y_true, y_predictd), recall_score(y_peaks, (y_predictd > t).astype(int)), precision_score(y_peaks, (y_predictd > t).astype(int))
scores[3] = roc_auc_score(y_peaks, y_avo), average_precision_score(y_peaks, y_avo), mse(y_true, y_avo), recall_score(y_peaks, (y_avo > t).astype(int)), precision_score(y_peaks, (y_avo > t).astype(int))
numpy.save("peaks/{}.chr{}.idx{}.npy".format(assay, chrom, count), scores)