import numpy
import sys
import os

def predict_RNAseq():
	celltypes = ['E003', 'E004', 'E005', 'E006', 'E007', 'E011', 'E012', 'E013', 'E016',
		'E017', 'E021', 'E022', 'E024', 'E027', 'E028', 'E035', 'E037', 'E038', 'E047',
		'E053', 'E054', 'E058', 'E061', 'E062', 'E063', 'E065', 'E066', 'E070', 'E071',
		'E079', 'E080', 'E087', 'E089', 'E090', 'E094', 'E095', 'E096', 'E097', 'E098',
		'E099', 'E104', 'E105', 'E106', 'E109', 'E112', 'E113', 'E119']

	n = 0
	with open('RNAseq_cmd.sh', 'w') as outfile:
		for i, celltype in enumerate(celltypes):
			print 'bash predict_task.sh'.format(celltype)
			outfile.write('bash predict_task.sh {}\n'.format(celltype))
			n += 1

	os.system('qsub -l mfree=2G,h_rt=1:00:00 -pe serial 8 -V -o $PWD/logs -e $PWD/logs -t 1-{}:1 ~/bin/generic_array_job.csh "RNAseq_cmd.sh"'.format(n))

predict_RNAseq()
