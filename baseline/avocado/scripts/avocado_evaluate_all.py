import os, glob

n_models = 0

with open("cmd.sh", "w") as outfile:
	for filename in glob.glob('models/*.h5'):
		outfilename = filename.replace('.h5', '.txt')
		if os.path.isfile(outfilename):
			continue		

		outfile.write('bash avocado_evaluate.sh {}\n'.format(filename))
		print "bash avocado_evaluate.sh {}".format(filename)
		n_models += 1

os.system('qsub -l mfree=8G,gpgpu=TRUE,cuda=1,h_rt=4:0:0 -V -o $PWD/logs -e $PWD/logs -t 1-{}:1 -tc 8 ~/bin/generic_array_job.csh "cmd.sh"'.format(n_models))
