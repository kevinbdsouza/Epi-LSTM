import os, sys   
os.environ['THEANO_FLAGS'] = "device=cuda{}".format(os.environ['SGE_HGR_cuda'])    
import theano

from avocado import *
from avocado.io import *
from avocado.models import *

batch_size = 40000
start, end = 0, 1126468

n_celltype, n_assay, n_position, n_histone, n_regulatory, n_layers, n_nodes = map(int, sys.argv[1:])

training_data = load_datasets(training_set)
validation_data = load_datasets(validation_set)

print len(training_data), len(validation_data)

X_train = sequential_data_generator(batch_size, start, end, training_data)
X_valid = data_generator(batch_size, start, end, validation_data)

model = HierarchicalAvocado(n_celltype, n_assay, n_position, n_histone, n_regulatory, n_layers, n_nodes, end-start)
model.summary()
model.fit_generator(X_train, 120, 200, workers=1, pickle_safe=True) #, validation_data=X_valid, validation_steps=30)
model.save("havocado_{}_{}_{}_{}_{}_{}_{}.h5".format(n_celltype, n_assay, n_position, n_histone, n_regulatory, n_layers, n_nodes))
