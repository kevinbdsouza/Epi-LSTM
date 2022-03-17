import numpy
import os

from avocado import *

n = 0
with open("peak_cmd.sh", "w") as outfile:
	for assay in assays:
		for i in range(1, 128):
			if os.path.isfile("peaks/{}.chr20.idx{}.npy".format(assay, i)):
				continue

			print "bash evaluate_peaks.sh {} {} {}".format(assay, 20, i)
			outfile.write("bash evaluate_peaks.sh {} {} {}\n".format(assay, 20, i))
			n += 1

os.system('qsub -l mfree=15G,h_rt=8:0:0 -V -o $PWD/logs -e $PWD/logs -t 1-{}:1 ~/bin/generic_array_job.csh "peak_cmd.sh"'.format(n))