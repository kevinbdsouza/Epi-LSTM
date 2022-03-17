import numpy
import sys
import xgboost

from avocado import *
from sklearn.metrics import average_precision_score

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

		model = xgboost.XGBClassifier(n_estimators=5000, nthread=min(X_train.shape[1], 8), max_depth=6)
		model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric='map', early_stopping_rounds=20, verbose=False)

		y_hat = model.predict_proba(X_test)
		maps[i] = average_precision_score(y_test, y_hat[:,1])

	return maps  


def celltype_baseline(y, celltypes, celltype):
	y_hat = numpy.zeros((celltypes == celltype).sum())
	y_hat[:] = y[celltypes == celltype].mean()
	y_test = y[celltypes == celltype]
	return average_precision_score(y_test, y_hat)

celltype = sys.argv[1]

X1 = numpy.load('datasets/RNAseq.x11.npy')
X2 = numpy.load('datasets/RNAseq.x2.npy')
X3 = numpy.load('datasets/RNAseq.x3.npy')
X4 = numpy.load('datasets/RNAseq.x4.npy')
X5 = numpy.load('datasets/RNAseq.x5.npy')[:,:110]
X6 = numpy.load('datasets/RNAseq.x6.npy')

y = numpy.load('datasets/RNAseq.npy')
y = (y >= 0.5).astype(int)
celltypes = numpy.load('datasets/RNAseq.celltypes.npy')

prs = numpy.zeros((7, 20))
prs[0] = make_predictions(X1, y, celltypes, celltype)
prs[1] = make_predictions(X2, y, celltypes, celltype)
prs[2] = make_predictions(X3, y, celltypes, celltype)
prs[3] = make_predictions(X4, y, celltypes, celltype)
prs[4] = make_predictions(X5, y, celltypes, celltype)
prs[5] = make_predictions(X6, y, celltypes, celltype)
prs[6] = celltype_baseline(y, celltypes, celltype)

numpy.save('outputs/RNAseq.{}.map20.npy'.format(celltype), prs)
