import os    
os.environ['THEANO_FLAGS'] = "device=cuda{}".format(os.environ['SGE_HGR_cuda'])    
import theano

from avocado import *
import keras

import sys, numpy, itertools, glob
numpy.random.seed(0)

numpy.set_printoptions(linewidth=300)

def predict_track(model, celltype, assay, start_coordinate, end_coordinate):
	n_positions = end_coordinate - start_coordinate

	celltype_idx = celltypes.index(celltype)
	assay_idx = assays.index(assay)

	celltype_idxs = numpy.ones(n_positions) * celltype_idx
	assay_idxs = numpy.ones(n_positions) * assay_idx

	genomic_idxs = numpy.arange(n_positions)
	histone_idxs = numpy.arange(n_positions) / 10
	regulatory_idxs = numpy.arange(n_positions) / 200

	X = {'celltype' : celltype_idxs, 'assay' : assay_idxs, 'genome' : genomic_idxs, 'histone' : histone_idxs, 'regulatory' : regulatory_idxs}
	y = model.predict(X, batch_size=40000)[:,0]
	return y

def mse(y_true, y_pred):
	return ((y_true - y_pred) ** 2).mean()

def evaluate_model(model):
	model_errors = numpy.zeros(100)
	start_coordinate, end_coordinate = 0, 1126468

	for i, (celltype, assay) in enumerate(validation_set):
		y_true = numpy.load('/net/noble/vol4/noble/user/jmschr/proj/avocado/data/{}.{}.pilot.arcsinh.npy'.format(celltype, assay))[start_coordinate:end_coordinate]
		y_pred = predict_track(model, celltype, assay, start_coordinate, end_coordinate)
		model_errors[i] = mse(y_true, y_pred)

	return model_errors

filename = sys.argv[1]
outfilename = filename.strip(".h5") + '.txt' 


with open(outfilename, "w") as outfile:
	model = keras.models.load_model(filename)
	errors = evaluate_model(model)

	params = filename.strip("models/havocado_").strip('.h5').split("_")
	outfile.write("\t".join(params) + "\t" + "\t".join(map(str, errors)))
	print errors[7:].mean()
