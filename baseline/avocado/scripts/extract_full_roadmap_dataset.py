import numpy
import sys

from avocado import *
from joblib import Parallel, delayed

data_dir = '/net/noble/vol4/noble/user/jmschr/proj/avocado/data'
chrom_dir = '/net/noble/vol5/user/jmschr/proj/avocado/predictions/ChromImpute'
pred_dir = '/net/noble/vol5/user/jmschr/proj/avocado/predictions/PREDICTD'
avo_dir = '/net/noble/vol5/user/jmschr/proj/avocado/predictions/avocado_full'

def load_track(data_dir, celltype, assay, chrom, starts, ends):
	x = []

	if 'ChromImpute' in data_dir:
		track = numpy.load('{}/{}.{}.chr{}.imputed.npz'.format(data_dir, celltype, assay, chrom))['arr_0']
	elif 'predictions/avocado_full' in data_dir:
		track = numpy.load('{}/{}.{}.chr{}.avocado.npz'.format(data_dir, celltype, assay, chrom))['arr_0']
	elif 'PREDICTD' in data_dir:
		track = numpy.load('{}/{}.{}.chr{}.predictd.npz'.format(data_dir, celltype, assay, chrom))['arr_0']
	else:
		track = numpy.load('{}/{}.{}.chr{}.arcsinh.npy'.format(data_dir, celltype, assay, chrom))

	for start, end in zip(starts, ends):
		if end >= track.shape[0]:
			value = 0
		else:
			value = track[start:end].mean()

		x.append(value)

	return numpy.array(x)

def load_all_epigenomic_data(chrom, starts, ends):
	all_tracks = training_set + validation_set + test_set
	X = Parallel(n_jobs=8)(delayed(load_track)(data_dir, celltype, assay, chrom, starts, ends) for celltype, assay in all_tracks)
	return numpy.array(X).T.copy()

chrom, name = sys.argv[1:]

starts = numpy.load("datasets/chr{}.{}.starts.npy".format(chrom, name))
ends = numpy.load("datasets/chr{}.{}.ends.npy".format(chrom, name))

x = load_all_epigenomic_data(chrom, starts, ends)

numpy.save("datasets/chr{}.{}.x6.npy".format(chrom, name), x)