import sys
import numpy
from sklearn.metrics import roc_auc_score
from avocado import *

data_dir = '/net/noble/vol4/noble/user/jmschr/proj/avocado/data'
pred_dir = '/net/noble/vol5/user/jmschr/proj/avocado/predictions'

def mse(y_true, y_pred):
	return ((y_true - y_pred) ** 2.).mean()

def mse1obs(y_true, y_pred):
	n = int(y_true.shape[0] * 0.01)
	y_true_sorted = numpy.sort(y_true)
	y_true_top1 = y_true_sorted[-n]
	idx = y_true >= y_true_top1
	return mse(y_true[idx], y_pred[idx])

def mse1imp(y_true, y_pred):
	n = int(y_true.shape[0] * 0.01)
	y_pred_sorted = numpy.sort(y_pred)
	y_pred_top1 = y_pred_sorted[-n]
	idx = y_pred >= y_pred_top1
	return mse(y_true[idx], y_pred[idx])

def gwcorr(y_true, y_pred):
	return numpy.corrcoef(y_true, y_pred)[0, 1]

def match1(y_true, y_pred):
	n = int(y_true.shape[0] * 0.01)
	y_true_sorted = numpy.sort(y_true)
	y_pred_sorted = numpy.sort(y_pred)
	
	y_true_top1 = y_true_sorted[-n]
	y_pred_top1 = y_pred_sorted[-n]

	y_true_top = y_true >= y_true_top1
	y_pred_top = y_pred >= y_pred_top1
	return (y_true_top & y_pred_top).sum()

def catch1obs(y_true, y_pred):
	n = int(y_true.shape[0] * 0.01)
	y_true_sorted = numpy.sort(y_true)
	y_pred_sorted = numpy.sort(y_pred)
	
	y_true_top1 = y_true_sorted[-n]
	y_pred_top1 = y_pred_sorted[-n*5]

	y_true_top = y_true >= y_true_top1
	y_pred_top = y_pred >= y_pred_top1
	return (y_true_top & y_pred_top).sum()

def catch1imp(y_true, y_pred):
	n = int(y_true.shape[0] * 0.01)
	y_true_sorted = numpy.sort(y_true)
	y_pred_sorted = numpy.sort(y_pred)
	
	y_true_top1 = y_true_sorted[-n*5]
	y_pred_top1 = y_pred_sorted[-n]

	y_true_top = y_true >= y_true_top1
	y_pred_top = y_pred >= y_pred_top1
	return (y_true_top & y_pred_top).sum()

def aucobs1(y_true, y_pred):
	n = int(y_true.shape[0] * 0.01)
	y_true_sorted = numpy.sort(y_true)
	y_true_top1 = y_true_sorted[-n]
	y_true_top = y_true >= y_true_top1
	return roc_auc_score(y_true_top, y_pred)

def aucimp1(y_true, y_pred):
	n = int(y_true.shape[0] * 0.01)
	y_pred_sorted = numpy.sort(y_pred)
	y_pred_top1 = y_pred_sorted[-n]
	y_pred_top = y_pred >= y_pred_top1
	return roc_auc_score(y_pred_top, y_true)

def mseprom(y_true, y_pred, chrom):
	sse, n = 0., 0.

	with open('gencode.v19.annotation.protein_coding.full.sorted.genes.bed', 'r') as infile:
		for line in infile:
			chrom_, start, end, _, _, strand = line.split()
			start = int(start) // 25
			end = int(end) // 25 + 1
			
			if chrom_ in ('chrX', 'chrY', 'chrM'):
				continue
				
			chrom_ = int(chrom_[3:])

			if chrom_ != chrom:
				continue

			if strand == '+':
				sse += ((y_true[start-80: start+1] - y_pred[start-80: start+1]) ** 2).sum()
				n += y_true[start-80:start+1].shape[0]

			else:
				sse += ((y_true[end: end+81] - y_pred[end: end+81]) ** 2).sum()
				n += y_true[end: end+81].shape[0]

	return sse / n

def msegene(y_true, y_pred, chrom):
	sse, n = 0., 0.

	with open('gencode.v19.annotation.protein_coding.full.sorted.genes.bed', 'r') as infile:
		for line in infile:
			chrom_, start, end, _, _, strand = line.split()
			start = int(start) // 25
			end = int(end) // 25 + 1
			
			if chrom_ in ('chrX', 'chrY', 'chrM'):
				continue
				
			chrom_ = int(chrom_[3:])

			if chrom_ != chrom:
				continue

			sse += ((y_true[start:end] - y_pred[start:end]) ** 2).sum()
			n += end - start

	return sse / n

def mseenh(y_true, y_pred, chrom):
	sse, n = 0., 0.

	with open('human_permissive_enhancers_phase_1_and_2.bed', 'r') as infile:
		for line in infile:
			chrom_, start, end, _, _, _, _, _, _, _, _, _ = line.split()
			start = int(start) // 25
			end = int(end) // 25 + 1
			
			if chrom_ in ('chrX', 'chrY', 'chrM'):
				continue
				
			chrom_ = int(chrom_[3:])

			if chrom_ != chrom:
				continue

			sse += ((y_true[start:end] - y_pred[start:end]) ** 2).sum()
			n += end - start

	return sse / n


celltype, assay = sys.argv[1:]

for chrom in range(1, 23):
	with open('results/{}.{}.chr{}.txt'.format(celltype, assay, chrom), 'w') as outfile:
		y_true = numpy.load('{}/{}.{}.chr{}.arcsinh.npy'.format(data_dir, celltype, assay, chrom))
		y_chrom = numpy.load('{}/ChromImpute/{}.{}.chr{}.imputed.npz'.format(pred_dir, celltype, assay, chrom))['arr_0']
		y_predictd = numpy.load('{}/PREDICTD/{}.{}.chr{}.predictd.npz'.format(pred_dir, celltype, assay, chrom))['arr_0']
		y_avo = numpy.load('{}/avocado/{}.{}.chr{}.avocado.npz'.format(pred_dir, celltype, assay, chrom))['arr_0']

		for func in mse, mse1obs, mse1imp, gwcorr, match1, catch1obs, catch1imp, aucobs1, aucimp1:
			outfile.write("{}\t{}\t{}\n".format(func(y_true, y_chrom), func(y_true, y_predictd), func(y_true, y_avo)))

		for func in mseprom, msegene, mseenh:
			outfile.write("{}\t{}\t{}\n".format(func(y_true, y_chrom, chrom), func(y_true, y_predictd, chrom), func(y_true, y_avo, chrom)))
