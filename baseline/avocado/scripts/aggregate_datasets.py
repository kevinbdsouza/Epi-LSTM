import numpy
import itertools

from avocado import *
from tqdm import tqdm

def aggregate_dataset(name, celltypes):
	x1, x2, x3, x4, x5, celltypez = [], [], [], [], [], []

	for celltype, chrom in tqdm(itertools.product(celltypes, range(1, 23))):
		print celltype, chrom
		x = numpy.load('datasets/{}.chr{}.{}.x1.npy'.format(celltype, chrom, name))
		x1.append(x)

		x = numpy.load('datasets/{}.chr{}.{}.x2.npy'.format(celltype, chrom, name))
		x2.append(x)

		x = numpy.load('datasets/{}.chr{}.{}.x3.npy'.format(celltype, chrom, name))
		x3.append(x)

		x = numpy.load('datasets/{}.chr{}.{}.x4.npy'.format(celltype, chrom, name))
		x4.append(x)

		x = numpy.load('datasets/{}.chr{}.{}.x5.npy'.format(celltype, chrom, name))
		x5.append(x)

		x = numpy.load('datasets/{}.chr{}.{}.celltypes.npy'.format(celltype, chrom, name))
		celltypez.append(x)

	x1 = numpy.concatenate(x1)
	x2 = numpy.concatenate(x2)
	x3 = numpy.concatenate(x3)
	x4 = numpy.concatenate(x4)
	x5 = numpy.concatenate(x5)
	celltypez = numpy.concatenate(celltypez)

	print x1.shape, x2.shape, x3.shape, x4.shape, x5.shape, celltypez.shape

	numpy.save("datasets/{}.x1.npy".format(name), x1)
	numpy.save("datasets/{}.x2.npy".format(name), x2)
	numpy.save("datasets/{}.x3.npy".format(name), x3)
	numpy.save("datasets/{}.x4.npy".format(name), x4)
	numpy.save("datasets/{}.x5.npy".format(name), x5)
	numpy.save("datasets/{}.celltypes.npy".format(name), celltypez)

RNAseq_celltypes = celltypes = ['E003', 'E004', 'E005', 'E006', 'E007', 'E011', 'E012', 'E013', 'E016',
		'E017', 'E021', 'E022', 'E024', 'E027', 'E028', 'E035', 'E037', 'E038', 'E047',
		'E053', 'E054', 'E058', 'E061', 'E062', 'E063', 'E065', 'E066', 'E070', 'E071',
		'E079', 'E080', 'E087', 'E089', 'E090', 'E094', 'E095', 'E096', 'E097', 'E098',
		'E099', 'E104', 'E105', 'E106', 'E109', 'E112', 'E113', 'E119']
TAD_celltypes = ['E116', 'E003', 'E017', 'E006', 'E004', 'E007', 'E005'] 

aggregate_dataset("TADs", TAD_celltypes)