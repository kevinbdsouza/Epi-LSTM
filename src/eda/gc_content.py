from Bio import SeqIO

fasta_dir = "/data2/latent/data/dna/"

chr = 21
gc_window = 100

fasta_path = fasta_dir + "chr" + str(chr) + ".fa"
records = list(SeqIO.parse(fasta_path, "fasta"))

gc_content = []
for cut in range(int(len(records[0]) / gc_window)):
    new_seq = records[0].seq[cut * gc_window: (cut + 1) * gc_window].lower()

    gc_count = 0
    for pos in range(gc_window):

        if new_seq[pos] == "g" or new_seq[pos] == "c":
            gc_count += gc_count

    gc_content.append(gc_count / gc_window)

# records[0].seq = new_seq
# SeqIO.write(records, "b1.fasta", "fasta")
# b1_read = list(SeqIO.parse("b1.fasta", "fasta"))

print("done")
