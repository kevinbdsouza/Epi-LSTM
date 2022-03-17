import numpy
import pandas
import itertools
import os

from avocado import *
from tqdm import tqdm

data = []

for celltype, chrom in tqdm(itertools.product(celltypes, range(1, 23))):
	if not os.path.isfile('genemarks/{}.chr{}.genemarks.csv'.format(celltype, chrom)):
		continue

	datum = pandas.read_csv("genemarks/{}.chr{}.genemarks.csv".format(celltype, chrom))
	data.append(datum)

data = pandas.concat(data)
data.to_csv("genemarks.csv")