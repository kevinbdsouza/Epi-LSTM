from avocado import *
import os

n = 0

with open('extract_cmd.sh', 'w') as outfile:
    for celltype in celltypes:
        for chrom in range(1, 23):
            print
            "bash extract_gene_data.sh {} {}".format(celltype, chrom)
            outfile.write("bash extract_gene_data.sh {} {}\n".format(celltype, chrom))
            n += 1

os.system(
    'qsub -l mfree=2G -V -o $PWD/logs -e $PWD/logs -t 1-{}:1 ~/bin/generic_array_job.csh "extract_cmd.sh"'.format(n))
