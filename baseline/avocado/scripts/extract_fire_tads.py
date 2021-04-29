import numpy
import pandas

from avocado import chromosome_lengths

celltypes = 'NPC', 'MES', 'MSC', 'TRO', 'GM12878', 'IMR90', 'H1'

for celltype in celltypes:
	fires = {'chr{}'.format(i+1): numpy.zeros(length // 1600 + 1) for i, length in enumerate(chromosome_lengths)}

	data = pandas.read_csv("Schmitt/{}_FIRE.tsv".format(celltype), sep="\t")
	for i, (chrom, start, end, fire) in data.iterrows():
		chrom, start, end, fire = int(chrom), int(start), int(end), float(fire)
		if chrom == 23:
			continue

		fires['chr{}'.format(chrom)][start // 40000: (end // 40000)] = 1

	for chrom, track in fires.items():
		numpy.save("Schmitt/{}_{}_FIRE.npy".format(celltype, chrom), track)


	tads = {'chr{}'.format(i+1): numpy.zeros(length // 1600 + 1) for i, length in enumerate(chromosome_lengths)}

	data = pandas.read_csv("Schmitt/{}_TADs.tsv".format(celltype), sep="\t")
	for i, (chrom, start, end) in data.iterrows():
		start, end = int(start), int(end)
		if chrom == 'chrX' or chrom == 'chrY' or chrom == 'chrM':
			continue

		tads[chrom][start // 40000: (end // 40000)] = 1

	for chrom, track in tads.items():
		numpy.save("Schmitt/{}_{}_TADs.npy".format(celltype, chrom), track)
