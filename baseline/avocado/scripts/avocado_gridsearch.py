import os, sys, numpy
import itertools
import glob
import time

n_celltype = 64, 128, 256
n_assay = 64, 128, 256
n_position = 5, 10, 15
n_histone = 20, 30
n_regulatory = 30, 45
n_layers = 2, 3
n_nodes = 128, 256, 512, 1024

n_celltype = 16, 32, 64, 128, 256
n_assay = 16, 32, 64, 128, 256
n_position = 5, 10, 15, 20, 25
n_histone = 10, 20, 30, 40, 50
n_regulatory = 15, 30, 45, 60 
n_layers = 0, 1, 2, 3, 4
n_nodes = 128, 256, 512, 1024, 2048

job = "qsub -l mfree=8G,gpgpu=TRUE,cuda=1,h_rt=4:0:0 -pe serial 3 avocado_fit.sh {} {} {} {} {} {} {}"
submitted = {}
n_submissions = int(sys.argv[1])
n_submitted = 0

for model in glob.glob('models/*.h5'):
	model = model.strip('models/havocado_').strip('.h5')
	params = tuple(map(int, model.split("_")))
	submitted[params] = True

cmd_file = open('cmd.sh', 'w')

while n_submitted < n_submissions:
	celltype = numpy.random.choice(n_celltype)
	assay = numpy.random.choice(n_assay)
	position = numpy.random.choice(n_position)
	histone = numpy.random.choice(n_histone)
	regulatory = numpy.random.choice(n_regulatory)
	layers = numpy.random.choice(n_layers)
	nodes = numpy.random.choice(n_nodes)

	index = (celltype, assay, position, histone, regulatory, layers, nodes)
	if index not in submitted:
		print job.format(celltype, assay, position, histone, regulatory, layers, nodes)

		cmd_file.write('bash avocado_fit.sh {} {} {} {} {} {} {}\n'.format(celltype, assay, position, histone, regulatory, layers, nodes))
		#os.system(job.format(celltype, assay, position, histone, regulatory, layers, nodes))
		
		submitted[index] = True
		n_submitted += 1
	else:
		print "Collision: ", index


os.system('qsub -l mfree=8G,gpgpu=TRUE,cuda=1,h_rt=4:0:0 -pe serial 3 -V -o $PWD/logs -e $PWD/logs -t 1-{}:1 -tc 7 ~/bin/generic_array_job.csh "cmd.sh"'.format(n_submissions))
cmd_file.close()
