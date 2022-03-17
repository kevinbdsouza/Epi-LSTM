import numpy
import itertools as it

from avocado import *
from tqdm import tqdm

from joblib import Parallel, delayed

import pandas

data_dir = '/net/noble/vol4/noble/user/jmschr/proj/avocado/data'
chrom_dir = '/net/noble/vol5/user/jmschr/proj/avocado/predictions/ChromImpute'
pred_dir = '/net/noble/vol5/user/jmschr/proj/avocado/predictions/PREDICTD'
avo_dir = '/net/noble/vol5/user/jmschr/proj/avocado/predictions/avocado_full'


def extract_RNAseq_assay(celltype, starts, ends):
    rna = []
    for chrom in range(1, 23):
        print
        chrom
        for start, end in zip(starts[chrom - 1], ends[chrom - 1]):
            x = numpy.load('{}/{}.RNA-seq.chr{}.arcsinh.npy'.format(data_dir, celltype, chrom), mmap_mode='r')

            if start > x.shape[0] or end > x.shape[0]:
                rna.append(0)
            else:
                rna.append(x[start:end].mean())

    return rna


def extract_RNAseq():
    histones = 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me3', 'H3K9me3'

    celltypes = ['E003', 'E004', 'E005', 'E006', 'E007', 'E011', 'E012', 'E013', 'E016',
                 'E017', 'E021', 'E022', 'E024', 'E027', 'E028', 'E035', 'E037', 'E038', 'E047',
                 'E053', 'E054', 'E058', 'E061', 'E062', 'E063', 'E065', 'E066', 'E070', 'E071',
                 'E079', 'E080', 'E087', 'E089', 'E090', 'E094', 'E095', 'E096', 'E097', 'E098',
                 'E099', 'E104', 'E105', 'E106', 'E109', 'E112', 'E113', 'E119']

    starts, ends = [[] for i in range(22)], [[] for i in range(22)]
    gene_starts, gene_ends = [[] for i in range(22)], [[] for i in range(22)]

    with open("gencode.v19.annotation.protein_coding.full.sorted.genes.bed", "r") as infile:
        for line in tqdm(infile):
            chrom, start, end, _, _, strand = line.split()
            start = int(start) // 25
            end = int(end) // 25 + 1

            if chrom in ('chrX', 'chrY', 'chrM'):
                continue

            chrom = int(chrom[3:])

            gene_starts[chrom - 1].append(start)
            gene_ends[chrom - 1].append(end + 1)

            if strand == '+':
                starts[chrom - 1].append(start - 80)
                ends[chrom - 1].append(start + 1)
            else:
                starts[chrom - 1].append(end)
                ends[chrom - 1].append(end + 81)

    numpy.save("datasets/RNAseq.histones.npy", histones)
    for i in range(22):
        start = numpy.array(starts[i])
        numpy.save("datasets/chr{}.RNAseq.starts.npy".format(i + 1), start)

        end = numpy.array(ends[i])
        numpy.save("datasets/chr{}.RNAseq.ends.npy".format(i + 1), end)

    n = 0
    with open('RNAseq_cmd.sh', 'w') as outfile:
        for celltype in celltypes:
            for chrom in range(1, 23):
                print
                'bash extract_dataset.sh {} {} RNAseq'.format(celltype, chrom)
                outfile.write('bash extract_dataset.sh {} {} RNAseq\n'.format(celltype, chrom))
                n += 1

    # os.system('qsub -l mfree=4G,h_rt=8:0:0 -V -o $PWD/logs -e $PWD/logs -t 1-{}:1 ~/bin/generic_array_job.csh "RNAseq_cmd.sh"'.format(n))

    RNAseq = Parallel(n_jobs=22)(
        delayed(extract_RNAseq_assay)(celltype, gene_starts, gene_ends) for celltype in celltypes)
    RNAseq = numpy.array(RNAseq).flatten()
    numpy.save("datasets/RNAseq2.npy", RNAseq)


def extract_TADs():
    histones = ['H3K27ac', 'H3K27me3', 'H3K36me3', 'H3K4me1', 'H3K4me3', 'H3K9ac', 'H3K9me3']
    celltypes = ['E116', 'E003', 'E017', 'E006', 'E004', 'E007', 'E005']

    starts, ends = [], []

    for chrom in range(1, 23):
        starts = numpy.arange(0, chromosome_lengths[chrom - 1], 1600)
        numpy.save("datasets/chr{}.TADs.starts.npy".format(chrom), starts)

        ends = starts + 1600
        numpy.save("datasets/chr{}.TADs.ends.npy".format(chrom), ends)

    numpy.save("datasets/TADs.histones.npy", histones)

    n = 0
    with open('TADs.cmd', 'w') as outfile:
        for celltype in celltypes:
            for chrom in range(1, 23):
                print
                'bash extract_dataset.sh {} {} TADs'.format(celltype, chrom)
                outfile.write('bash extract_dataset.sh {} {} TADs\n'.format(celltype, chrom))
                n += 1

    # os.system('qsub -l mfree=4G,h_rt=8:0:0 -V -o $PWD/logs -e $PWD/logs -t 1-{}:1 ~/bin/generic_array_job.csh "TADs.cmd"'.format(n))

    n = 0
    with open('TADs.all.cmd', 'w') as outfile:
        for chrom in range(1, 23):
            print
            "bash extract_full_roadmap_dataset.sh {} TADs".format(chrom)
            outfile.write("bash extract_full_roadmap_dataset.sh {} TADs\n".format(chrom))
            n += 1

    os.system(
        'qsub -l mfree=2G,h_rt=8:0:0 -pe serial 8 -V -o $PWD/logs -e $PWD/logs -t 1-{}:1 ~/bin/generic_array_job.csh "TADs.all.cmd"'.format(
            n))


extract_RNAseq()
# extract_TADs()
