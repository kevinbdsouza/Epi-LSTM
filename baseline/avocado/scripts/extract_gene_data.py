import numpy
import sys

from avocado import *
from tqdm import tqdm

import pandas

data_dir = '/net/noble/vol4/noble/user/jmschr/proj/avocado/data'
pred_dir = '/net/noble/vol5/user/jmschr/proj/avocado/predictions'


def load_chromosome(celltype, chrom):
    RNAseq = []
    H3K36me3_1, H3K36me3_2, H3K36me3_3, H3K36me3_4 = [], [], [], []
    H3K27me3_1, H3K27me3_2, H3K27me3_3, H3K27me3_4 = [], [], [], []
    H3K4me3_1, H3K4me3_2, H3K4me3_3, H3K4me3_4 = [], [], [], []
    H3K4me1_1, H3K4me1_2, H3K4me1_3, H3K4me1_4 = [], [], [], []
    corr_1, corr_2, corr_3, corr_4 = [], [], [], []
    starts, ends = [], []

    a1 = numpy.load(data_dir + "/{}.H3K4me3.chr{}.arcsinh.npy".format(celltype, chrom), mmap_mode='r')
    b1 = numpy.load(data_dir + "/{}.H3K27me3.chr{}.arcsinh.npy".format(celltype, chrom), mmap_mode='r')
    c1 = numpy.load(data_dir + "/{}.H3K36me3.chr{}.arcsinh.npy".format(celltype, chrom), mmap_mode='r')
    d1 = numpy.load(data_dir + "/{}.H3K4me1.chr{}.arcsinh.npy".format(celltype, chrom), mmap_mode='r')

    a2 = numpy.load(pred_dir + "/ChromImpute/{}.H3K4me3.chr{}.imputed.npz".format(celltype, chrom), mmap_mode='r')[
        'arr_0']
    b2 = numpy.load(pred_dir + "/ChromImpute/{}.H3K27me3.chr{}.imputed.npz".format(celltype, chrom), mmap_mode='r')[
        'arr_0']
    c2 = numpy.load(pred_dir + "/ChromImpute/{}.H3K36me3.chr{}.imputed.npz".format(celltype, chrom), mmap_mode='r')[
        'arr_0']
    d2 = numpy.load(pred_dir + "/ChromImpute/{}.H3K4me1.chr{}.imputed.npz".format(celltype, chrom), mmap_mode='r')[
        'arr_0']

    a3 = numpy.load(pred_dir + "/PREDICTD/{}.H3K4me3.chr{}.predictd.npz".format(celltype, chrom))['arr_0']
    b3 = numpy.load(pred_dir + "/PREDICTD/{}.H3K27me3.chr{}.predictd.npz".format(celltype, chrom))['arr_0']
    c3 = numpy.load(pred_dir + "/PREDICTD/{}.H3K36me3.chr{}.predictd.npz".format(celltype, chrom))['arr_0']
    d3 = numpy.load(pred_dir + "/PREDICTD/{}.H3K4me1.chr{}.predictd.npz".format(celltype, chrom))['arr_0']

    a4 = numpy.load(pred_dir + "/avocado/{}.H3K4me3.chr{}.avocado.npz".format(celltype, chrom), mmap_mode='r')['arr_0']
    b4 = numpy.load(pred_dir + "/avocado/{}.H3K27me3.chr{}.avocado.npz".format(celltype, chrom), mmap_mode='r')['arr_0']
    c4 = numpy.load(pred_dir + "/avocado/{}.H3K36me3.chr{}.avocado.npz".format(celltype, chrom), mmap_mode='r')['arr_0']
    d4 = numpy.load(pred_dir + "/avocado/{}.H3K4me1.chr{}.avocado.npz".format(celltype, chrom), mmap_mode='r')['arr_0']

    try:
        c = numpy.load(data_dir + "/{}.RNA-seq.chr{}.arcsinh.npy".format(celltype, chrom), mmap_mode='r')
        rnaseq = numpy.zeros_like(c4)
        rnaseq[:len(c)] = c
    except:
        rnaseq = numpy.zeros_like(c4) - 1

    with open('gencode.v19.annotation.protein_coding.full.sorted.genes.bed', 'r') as infile:
        for i, line in tqdm(enumerate(infile)):
            chrom_, start, end, _, _, strand = line.split()
            start = int(start) // 25
            end = int(end) // 25 + 1

            if chrom_ in ('chrX', 'chrY', 'chrM'):
                continue

            chrom_ = int(chrom_[3:])

            if chrom_ != chrom:
                continue

            starts.append(start)
            ends.append(end)

            RNAseq.append(rnaseq[start:end].mean())

            H3K36me3_1.append(c1[start:end].mean())
            H3K36me3_2.append(c2[start:end].mean())
            H3K36me3_3.append(c3[start:end].mean())
            H3K36me3_4.append(c4[start:end].mean())

            if strand == '+':
                a, b, d = a1[start - 80:start + 1], b1[start - 80:start + 1], d1[start - 80:start + 1]
                H3K4me3_1.append(a.mean())
                H3K27me3_1.append(b.mean())
                H3K4me1_1.append(d.mean())
                corr_1.append(numpy.corrcoef(a, b)[0, 1])

                a, b, d = a2[start - 80:start + 1], b2[start - 80:start + 1], d2[start - 80:start + 1]
                H3K4me3_2.append(a.mean())
                H3K27me3_2.append(b.mean())
                H3K4me1_2.append(d.mean())
                corr_2.append(numpy.corrcoef(a, b)[0, 1])

                a, b, d = a3[start - 80:start + 1], b3[start - 80:start + 1], d3[start - 80:start + 1]
                H3K4me3_3.append(a.mean())
                H3K27me3_3.append(b.mean())
                H3K4me1_3.append(d.mean())
                corr_3.append(numpy.corrcoef(a, b)[0, 1])

                a, b, d = a4[start - 80:start + 1], b4[start - 80:start + 1], d4[start - 80:start + 1]
                H3K4me3_4.append(a.mean())
                H3K27me3_4.append(b.mean())
                H3K4me1_4.append(d.mean())
                corr_4.append(numpy.corrcoef(a, b)[0, 1])

            else:
                a, b, d = a1[end:end + 81], b1[end:end + 81], d1[end:end + 81]
                H3K4me3_1.append(a.mean())
                H3K27me3_1.append(b.mean())
                H3K4me1_1.append(d.mean())
                corr_1.append(numpy.corrcoef(a, b)[0, 1])

                a, b, d = a2[end:end + 81], b2[end:end + 81], d2[end:end + 81]
                H3K4me3_2.append(a.mean())
                H3K27me3_2.append(b.mean())
                H3K4me1_2.append(d.mean())
                corr_2.append(numpy.corrcoef(a, b)[0, 1])

                a, b, d = a3[end:end + 81], b3[end:end + 81], d3[end:end + 81]
                H3K4me3_3.append(a.mean())
                H3K27me3_3.append(b.mean())
                H3K4me1_3.append(d.mean())
                corr_3.append(numpy.corrcoef(a, b)[0, 1])

                a, b, d = a4[end:end + 81], b4[end:end + 81], d4[end:end + 81]
                H3K4me3_4.append(a.mean())
                H3K27me3_4.append(b.mean())
                H3K4me1_4.append(d.mean())
                corr_4.append(numpy.corrcoef(a, b)[0, 1])

    celltype_idx = numpy.ones_like(corr_1) * celltypes.index(celltype)
    chrom_idx = numpy.ones_like(corr_1) * chrom

    data = {'celltype': celltype_idx, 'chrom': chrom_idx, 'RNAseq': RNAseq, 'H3K36me3_Roadmap': H3K36me3_1,
            'H3K36me3_ChromImpute': H3K36me3_2, 'H3K36me3_PREDICTD': H3K36me3_3,
            'H3K36me3_Avocado': H3K36me3_4, 'H3K27me3_Roadmap': H3K27me3_1, 'H3K27me3_ChromImpute': H3K27me3_2,
            'H3K27me3_PREDICTD': H3K27me3_3, 'H3K27me3_Avocado': H3K27me3_4,
            'H3K4me3_Roadmap': H3K4me3_1, 'H3K4me3_ChromImpute': H3K4me3_2, 'H3K4me3_PREDICTD': H3K4me3_3,
            'H3K4me3_Avocado': H3K4me3_4, 'H3K4me1_Roadmap': H3K4me1_1,
            'H3K4me1_ChromImpute': H3K4me1_2, 'H3K4me1_PREDICTD': H3K4me1_3, 'H3K4me1_Avocado': H3K4me1_4,
            'Roadmap_corr': corr_1, 'ChromImpute_corr': corr_2, 'PREDICTD_corr': corr_3,
            'Avocado_corr': corr_4, 'start': starts, 'end': ends}

    data = pandas.DataFrame(data)
    return data


celltype, chrom = sys.argv[1:]
chrom = int(chrom)

data = load_chromosome(celltype, chrom)
print
data.columns
data.to_csv("genemarks/{}.chr{}.genemarks.csv".format(celltype, chrom))
