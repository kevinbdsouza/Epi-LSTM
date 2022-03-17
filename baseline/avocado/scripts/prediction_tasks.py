import numpy
import pandas
import itertools as it
import xgboost
import time

from avocado import *
from avocado import celltypes as roadmap_celltypes
all_tracks = training_set + validation_set + test_set

from sklearn.metrics import average_precision_score

from joblib import Parallel, delayed
from tqdm import tqdm
import shap

def load_track(data_dir, celltype, assay, chrom, starts, ends):
	x = []

	if 'ChromImpute' in data_dir:
		track = numpy.load('{}/{}.{}.chr{}.imputed.npz'.format(data_dir, celltype, assay, chrom), mmap_mode='r')['arr_0']
	elif 'predictions/avocado_full' in data_dir:
		track = numpy.load('{}/{}.{}.chr{}.avocado.npz'.format(data_dir, celltype, assay, chrom), mmap_mode='r')['arr_0']
	elif 'PREDICTD' in data_dir:
		track = numpy.load('{}/{}.{}.chr{}.predictd.npz'.format(data_dir, celltype, assay, chrom), mmap_mode='r')['arr_0']
	else:
		track = numpy.load('{}/{}.{}.chr{}.arcsinh.npy'.format(data_dir, celltype, assay, chrom), mmap_mode='r')

	for start, end in zip(starts, ends):
		if end >= track.shape[0]:
			value = 0
		else:
			value = track[start:end].mean()

		x.append(value)

	return numpy.array(x)

def load_epigenomic_data(data_dir, celltype, assays, chrom, starts, ends):
	X = [load_track(data_dir, celltype, assay, chrom, starts, ends) for assay in tqdm(assays)]
	return numpy.array(X).T.copy()

def load_all_epigenomic_data(chrom, starts, ends):
	all_tracks = training_set + validation_set + test_set
	X = Parallel(n_jobs=20)(delayed(load_track)(data_dir, celltype, assay, chrom, starts, ends) for celltype, assay in tqdm(all_tracks))
	return numpy.array(X).T.copy()

def load_dataset(celltype, histones, chrom, starts, ends):
	x1 = load_epigenomic_data(data_dir, celltype, histones, chrom, starts, ends)
	x2 = load_epigenomic_data(chrom_dir, celltype, assays, chrom, starts, ends)
	x3 = load_epigenomic_data(pred_dir, celltype, assays, chrom, starts, ends)
	x4 = load_epigenomic_data(avo_dir, celltype, assays, chrom, starts, ends)
	celltypes = [celltype]*len(x1)
	return x1, x2, x3, x4, celltypes

def load_datasets(cell_types, histones, chrom, starts, ends):
	data = Parallel(n_jobs=8)(delayed(load_dataset)(celltype, histones, chrom, starts, ends) for celltype in cell_types)
	X1 = numpy.concatenate([datum[0] for datum in data])
	X2 = numpy.concatenate([datum[1] for datum in data])
	X3 = numpy.concatenate([datum[2] for datum in data])
	X4 = numpy.concatenate([datum[3] for datum in data])

	X5 = load_factors(chrom, starts, ends)
	X5 = numpy.concatenate([X5 for ct in cell_types])
	celltypes = numpy.concatenate([datum[4] for datum in data])
	return X1, X2, X3, X4, X5, celltypes

def load_factors(chrom, starts, ends):
	X = []

	genome_embedding = numpy.load('../1_12_2018_Full_Model/genome_embedding_chr{}.npy'.format(chrom), mmap_mode='r')

	for start, end in tqdm(zip(starts, ends)):
		factors = genome_embedding[start:end].mean(axis=0)
		X.append(factors)

	return numpy.array(X)

def make_predictions(X, y, celltypes, celltype):
	numpy.random.seed(0)

	X = X[celltypes == celltype]
	y = y[celltypes == celltype]

	idx = numpy.arange(X.shape[0])
	numpy.random.shuffle(idx)
	X, y = X[idx], y[idx]

	n_folds = 20
	maps = numpy.zeros(n_folds)
	for i in range(n_folds):
		X_test = X[i::n_folds]
		X_valid = X[(i+1)%n_folds::n_folds]
		X_train = numpy.concatenate([X[j::n_folds] for j in range(n_folds) if j != i and j != (i+1)%n_folds])

		y_test = y[i::n_folds]
		y_valid = y[(i+1)%n_folds::n_folds]
		y_train = numpy.concatenate([y[j::n_folds] for j in range(n_folds) if j != i and j != (i+1)%n_folds])

		model = xgboost.XGBClassifier(n_estimators=5000, nthread=min(X_train.shape[1], 20), max_depth=6)
		model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='map', early_stopping_rounds=20, verbose=False)

		y_hat = model.predict_proba(X_test)
		maps[i] = average_precision_score(y_test, y_hat[:,1])

	return maps  

def celltype_baseline(y, celltypes, celltype):
	y_hat = numpy.zeros((celltypes == celltype).sum())
	y_hat[:] = y[celltypes == celltype].mean()
	y_test = y[celltypes == celltype]
	return average_precision_score(y_test, y_hat)	

data_dir = '/net/noble/vol4/noble/user/jmschr/proj/avocado/data'
chrom_dir = '/net/noble/vol5/user/jmschr/proj/avocado/predictions/ChromImpute'
pred_dir = '/net/noble/vol5/user/jmschr/proj/avocado/predictions/PREDICTD'
avo_dir = '/net/noble/vol5/user/jmschr/proj/avocado/predictions/avocado_full'

def run_TADs():
	celltype_idx = ('GM12878', 'E116'), ('H1', 'E003'), ('IMR90', 'E017'), ('MSC', 'E006'), ('MES', 'E004'), ('NPC', 'E007'), ('TRO', 'E005')
	celltypes_ = ['E116', 'E003', 'E017', 'E006', 'E004', 'E007', 'E005']  
	celltypes = numpy.load("datasets/TADs.celltypes.npy")

	TADs = []
	for (celltype, _), chrom in it.product(celltype_idx, range(1, 23)):
		TADs.append(numpy.load('Schmitt/{}_chr{}_TADs.npy'.format(celltype, chrom)))
	TADs = numpy.concatenate(TADs)

	X1 = numpy.load("datasets/TADs.x11.npy")
	X2 = numpy.load("datasets/TADs.x2.npy")
	X3 = numpy.load("datasets/TADs.x3.npy")
	X4 = numpy.load("datasets/TADs.x4.npy")
	X5 = numpy.load("datasets/TADs.x5.npy")[:,:110]
	X6 = numpy.concatenate([numpy.load("datasets/chr{}.TADs.x6.npy".format(chrom)) for chrom in range(1, 23)])
	X6 = numpy.concatenate([X6 for i in range(7)])

	prs = numpy.zeros((7, 7, 20))
	for idx, celltype in zip(range(7), celltypes_):
		for j, X in enumerate([X1, X2, X3, X4, X5, X6]):
			prs[idx, j] = make_predictions(X, TADs, celltypes, celltype)

		prs[idx, 6] = celltype_baseline(TADs, celltypes, celltype)
		print celltype, prs[idx].mean(axis=1)

	numpy.save("TAD_maps20.npy", prs)


def run_FIREs():
	celltype_idx = ('GM12878', 'E116'), ('H1', 'E003'), ('IMR90', 'E017'), ('MSC', 'E006'), ('MES', 'E004'), ('NPC', 'E007'), ('TRO', 'E005')
	celltypes_ = ['E116', 'E003', 'E017', 'E006', 'E004', 'E007', 'E005']
	celltypes = numpy.load("datasets/TADs.celltypes.npy")

	FIREs = []
	for (celltype, _), chrom in it.product(celltype_idx, range(1, 23)):
		FIRE = numpy.load('Schmitt/{}_chr{}_FIRE.npy'.format(celltype, chrom))
		FIREs.append(FIRE)
	FIREs = numpy.concatenate(FIREs)
	FIREs = (FIREs > 0).astype(int)
	print FIREs.max(), FIREs.mean()

	X1 = numpy.load("datasets/TADs.x11.npy")
	X2 = numpy.load("datasets/TADs.x2.npy")
	X3 = numpy.load("datasets/TADs.x3.npy")
	X4 = numpy.load("datasets/TADs.x4.npy")
	X5 = numpy.load("datasets/TADs.x5.npy")[:,:110]
	X6 = numpy.concatenate([numpy.load("datasets/chr{}.TADs.x6.npy".format(chrom)) for chrom in range(1, 23)])
	X6 = numpy.concatenate([X6 for i in range(7)])

	prs = numpy.zeros((7, 7, 20))
	for idx, celltype in zip(range(7), celltypes_):
		results = []
		for j, X in enumerate([X1, X2, X3, X4, X5, X6]):
			prs[idx, j] = make_predictions(X, FIREs, celltypes, celltype)

		prs[idx, 6] = celltype_baseline(FIREs, celltypes, celltype)
		print celltype, prs[idx].mean(axis=1)

	numpy.save("FIRE_maps20.npy", prs)

def run_PEIs():
	X1 = numpy.load("pei_X1.npy")
	X2 = numpy.load("pei_X2.npy")
	X3 = numpy.load("pei_X3.npy")
	X4 = numpy.load("pei_X4.npy")
	X5 = numpy.load("pei_X5.npy")
	X6 = numpy.load("pei_X6.npy")
	X7 = numpy.load("pei_tf_X7.npy")
	X8 = numpy.load("pei_tf_X8.npy")
	
	y = numpy.load("pei_y.npy")
	y2 = numpy.load("pei_tf_y.npy")
	celltypes = numpy.load("pei_celltype.npy")
	celltypes2 = numpy.load("pei_tf_celltype.npy")

	celltypes_ = ['E017', 'E116', 'E117', 'E123']

	prs = numpy.zeros((4, 8, 20))
	for idx, celltype in zip(range(7), celltypes_):
		for j, X in enumerate([X1, X2, X3, X4, X5, X6]):
			prs[idx, j] = make_predictions(X, y, celltypes, celltype)

		prs[idx, 6] = make_predictions(X7, y2, celltypes2, celltype)
		prs[idx, 7] = make_predictions(X8, y2, celltypes2, celltype)
		print celltype, prs[idx].mean(axis=1)

	numpy.save("pei_maps20.npy", prs)
	print prs

def explain_TADs():
	#fires = pandas.read_excel('mmc6.xlsx')[['chr', 'start', 'end', 'GM12878', 'H1', 'IMR90', 'MSC', 'MES', 'NPC', 'TRO']]
	celltype_idx = ('GM12878', 'E116'), ('H1', 'E003'), ('IMR90', 'E017'), ('MSC', 'E006'), ('MES', 'E004'), ('NPC', 'E007'), ('TRO', 'E005')
	histones = ['H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me3', 'H3K9ac', 'H3K9me3']
	n = len(celltype_idx)

	celltypes_ = ['E116', 'E003', 'E017', 'E006', 'E004', 'E007', 'E005']  

	TADs = []
	for (celltype, _), chrom in it.product(celltype_idx, range(1, 23)):
		TADs.append(numpy.load('Schmitt/{}_chr{}_TADs.npy'.format(celltype, chrom)))

	TADs = numpy.concatenate(TADs)

	print "loading data"

	celltypes = numpy.load("datasets/TADs.celltypes.npy")
	X1 = numpy.load("datasets/TADs.x11.npy")

	shap_matrix = numpy.zeros((7, 24))

	for i, celltype in enumerate(celltypes_):
		X = X1[celltypes == celltype]
		y = TADs[celltypes == celltype]

		idx = numpy.arange(X.shape[0])
		numpy.random.shuffle(idx)
		X, y = X[idx], y[idx]

		X_valid, y_valid = X[::5], y[::5]
		X_train = numpy.concatenate([X[j::5] for j in range(1, 5)])
		y_train = numpy.concatenate([y[j::5] for j in range(1, 5)])

		model = xgboost.XGBClassifier(n_estimators=5000, nthread=min(X_train.shape[1], 20), max_depth=6)
		model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='map', early_stopping_rounds=20, verbose=None)

		shap_values = numpy.abs(shap.TreeExplainer(model).shap_values(X)).mean(axis=0)[:-1]
		shap_matrix[i] = shap_values

		print celltype, numpy.around(shap_values, 3)

	numpy.save("TADs.shap_values2.npy", shap_matrix)

def explain_FIREs():
	#fires = pandas.read_excel('mmc6.xlsx')[['chr', 'start', 'end', 'GM12878', 'H1', 'IMR90', 'MSC', 'MES', 'NPC', 'TRO']]
	celltype_idx = ('GM12878', 'E116'), ('H1', 'E003'), ('IMR90', 'E017'), ('MSC', 'E006'), ('MES', 'E004'), ('NPC', 'E007'), ('TRO', 'E005')
	histones = ['H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me3', 'H3K9ac', 'H3K9me3']
	n = len(celltype_idx)

	celltypes_ = ['E116', 'E003', 'E017', 'E006', 'E004', 'E007', 'E005']  

	FIREs = []
	for (celltype, _), chrom in it.product(celltype_idx, range(1, 23)):
		FIRE = numpy.load('Schmitt/{}_chr{}_FIRE.npy'.format(celltype, chrom))
		FIREs.append(FIRE)

	FIREs = numpy.concatenate(FIREs)
	FIREs = (FIREs > 0).astype(int)

	print "loading data"

	celltypes = numpy.load("datasets/TADs.celltypes.npy")
	X1 = numpy.load("datasets/TADs.x11.npy")

	shap_matrix = numpy.zeros((7, 24)) - 1

	for i, celltype in enumerate(celltypes_):
		X = X1[celltypes == celltype]
		y = FIREs[celltypes == celltype]

		idx = numpy.arange(X.shape[0])
		numpy.random.shuffle(idx)
		X, y = X[idx], y[idx]

		X_valid, y_valid = X[::5], y[::5]
		X_train = numpy.concatenate([X[j::5] for j in range(1, 5)])
		y_train = numpy.concatenate([y[j::5] for j in range(1, 5)])

		model = xgboost.XGBClassifier(n_estimators=5000, nthread=min(X_train.shape[1], 20), max_depth=6)
		model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='map', early_stopping_rounds=20, verbose=None)

		shap_values = numpy.abs(shap.TreeExplainer(model).shap_values(X)).mean(axis=0)[:-1]
		shap_matrix[i] = shap_values

		print celltype, numpy.around(shap_values, 3)

	numpy.save("FIREs.shap_values2.npy", shap_matrix)

#explain_TADs()
#explain_FIREs()

#run_FIREs()
#run_TADs()
run_PEIs()
