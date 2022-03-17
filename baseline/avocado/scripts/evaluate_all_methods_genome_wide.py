import os
from avocado import *

with open('evaluate_genomewide_cmd.sh', 'w') as outfile:
	with open('../../scripts/ExistingRoadmapTracks.txt', 'r') as infile:
		for line in infile:
			celltype, assay = line.split()

			print "bash evaluate_methods_genome_wide.sh {} {}".format(celltype, assay)
			outfile.write("bash evaluate_methods_genome_wide.sh {} {}\n".format(celltype, assay))

os.system('qsub -l mfree=5G,h_rt=8:0:0 -V -o $PWD/logs -e $PWD/logs -t 1-1014:1 ~/bin/generic_array_job.csh "evaluate_genomewide_cmd.sh"')
