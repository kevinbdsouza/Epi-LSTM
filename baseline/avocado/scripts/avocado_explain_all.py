import os

n = 0
with open('explain_cmd.sh', 'w') as outfile:
	for i in range(50):
		outfile.write("bash avocado_explain.sh 20 {}\n".format(i))
		n += 1
		
os.system('qsub -l mfree=20G,gpgpu=TRUE,cuda=1,cuda_gen_slow=20 -soft -l cuda_gen_slow=10 -V -o $PWD/logs -e $PWD/logs -t 1-{}:1 -tc 5 ~/bin/generic_array_job.csh "explain_cmd.sh"'.format(n))
