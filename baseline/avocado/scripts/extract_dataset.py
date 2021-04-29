import numpy
import sys
from avocado import *
from avocado import celltypes as roadmap_celltypes

data_dir = '/net/noble/vol4/noble/user/jmschr/proj/avocado/data'
chrom_dir = '/net/noble/vol5/user/jmschr/proj/avocado/predictions/ChromImpute'
pred_dir = '/net/noble/vol5/user/jmschr/proj/avocado/predictions/PREDICTD'
avo_dir = '/net/noble/vol5/user/jmschr/proj/avocado/predictions/avocado_full'

def load_epigenomic_data(data_dir, celltype, assays, chrom, starts, ends):
	X = []

	for assay in assays:
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

		X.append(x)

	return numpy.array(X).T.copy()

def load_factors(celltype, chrom, starts, ends):
	X = []

	genome_embedding = numpy.load('../1_12_2018_Full_Model/genome_embedding_chr{}.npy'.format(chrom))
	celltype_embedding = numpy.load('../1_12_2018_Full_Model/celltype_embedding.npy')

	cell_type_idx = roadmap_celltypes.index(celltype)

	for start, end in zip(starts, ends):
		factors = numpy.concatenate([genome_embedding[start:end].mean(axis=0), celltype_embedding[cell_type_idx]])
		X.append(factors)

	return numpy.array(X)

def load_dataset(celltype, histones, chrom, starts, ends):
	x1 = load_epigenomic_data(data_dir, celltype, histones, chrom, starts, ends)
	x2 = load_epigenomic_data(chrom_dir, celltype, assays, chrom, starts, ends)
	x3 = load_epigenomic_data(pred_dir, celltype, assays, chrom, starts, ends)
	x4 = load_epigenomic_data(avo_dir, celltype, assays, chrom, starts, ends)
	x5 = load_factors(celltype, chrom, starts, ends)
	celltypes = numpy.array([celltype]*len(x1))
	return x1, x2, x3, x4, x5, celltypes

celltype, chrom, name = sys.argv[1:]

starts = numpy.load("datasets/chr{}.{}.starts.npy".format(chrom, name))
ends = numpy.load("datasets/chr{}.{}.ends.npy".format(chrom, name))
histones = numpy.load("datasets/{}.histones.npy".format(name))

x1, x2, x3, x4, x5, celltypes = load_dataset(celltype, histones, chrom, starts, ends)

numpy.save("datasets/{}.chr{}.{}.x1.npy".format(celltype, chrom, name), x1)
numpy.save("datasets/{}.chr{}.{}.x2.npy".format(celltype, chrom, name), x2)
numpy.save("datasets/{}.chr{}.{}.x3.npy".format(celltype, chrom, name), x3)
numpy.save("datasets/{}.chr{}.{}.x4.npy".format(celltype, chrom, name), x4)
numpy.save("datasets/{}.chr{}.{}.x5.npy".format(celltype, chrom, name), x5)
numpy.save("datasets/{}.chr{}.{}.celltypes.npy".format(celltype, chrom, name), celltypes)
