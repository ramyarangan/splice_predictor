"""
Assemble full RNA sequences from the provided dataset of a library of DNA constructs.

Collate corresponding splicing efficiency values (ranging from 0 to 1) for each of 
four replicates with different index sequences.

Assemble information useful for baseline model: the indices of the 5' splice site, branchpoint, and 3' splice site

Notes on library construction in data/sequence_notes.txt

Example usage: python get_sequences.py ../data/SplicingLibData_Indices.csv ../data/splicing_data.csv
"""

import csv
import sys
import numpy as np
import pandas as pd

VERBOSE = False 

csv_filename = sys.argv[1] # "../data/SplicingLibData_Indices.csv"
data_filename = sys.argv[2] # "../data/splicing_data.csv"

# This background sequence includes the full promoter, the full mutated MUD1 gene, and terminator
background_seq_1 = "GGGGACCAGGTGCCGTAAGCGCTGCGGCAAAGCCGGACAGGCAGCGACAGCCCTGACAGACAAGACTCTCCTAGCTGCATGACTCATTTCACCGGTCGCGTTCGCCGGAACCGGCTTTTTTTTTTTTTTTTCGCGATCCTAGGGCGATCAATTGGCATTATCACATAATGAATTATACATTATATAAAGTAATGTGATTTCTTCGAAGAATATACTAAAAAATGAGCAGGCAAGATAAACGAAGGCAAAGGAAACAAAAAAAAAAAAAGTGGAAGTCAGGGTGTTGGTGTAAAGTCAATGTGACTGCGTTCCACTCCCCACAATAAATTACAAATATCATCACACCCGCTGATCTTACTAGGTGATCAACTGGGCCTTCTTGAAGTTTCTATTTCAAGATCATCAAAGCTGACTAACCAAGCATTTTTAACGTTTGTCACTCAAGAAGAAGCAGACCGGTTTCTAGAAAAATACACGACAACAGCATTAAAAGTTCAAGGTCGTAAAGTGAGACTGGGAAAGACTCGAACAAATTCGTTATTGGGTCATTCAATAGAAATCAAAAAAAAAAAAGGTAATTACGAAACGTACAACCTTGATATAAAGAAGGTGCTTAAAGCAAGGAAACTTAAACGCAAGTTACGTAGTGGTGATATTTGCGCCACAAAGTTCAGGCTTAAAAGGCAAATAAGACGCGTGAAGCATAAGCTGAGATCAAGAAAGGTAGAGGAGGCTGAGAGTGATAGGATTGTAAAAGAATTTGAAACCCGTAGATTCGAAAACATCAAGTCTCAACCAGAAAATCTAAAACAATCGCAGAAACCTCTTAAGCGGGCTAAAGTGTCCAATACAAAGGAAAATACACCGAACAAGGTCCTTCTTATACAAAATTTGCCTAGCGGCACTACCGAGCAATTATCCTCGCAAATACTTGGCAAGGAGGTTTTAGTTGAAATCAGTTTAGTTAGCGTTCGTTACCTAGCTTTCGTGGAATACGAGACCGTTGCTGTTGCTACGAAAATCAAGAATCAGTTAGGCTCCACTTACAAGCTACAAAACAATAACGTTACCATAGGATTTGCTAAGTAGAATATCCTTTGCGGAAGTATACCTCGAGTAAAGAAATTCACAGATAAATTTGAATAACGTTCTCCATTATTAATAAATATTTATTTACAAGCTTAGAAAATAAGGTGCTCTTTATTTACCGCGAATTTCTTATGATTTATGATTTTTATTATTAAATAAGTTATAAAAAAAATAAGTGTATACAAATTTTAAAGTGACTCTTAGGTTTTAAAACGAAAATTCTTATTCTTGAGTAACTCTTTCCTGTAGGTCAGGTTGCTTTCTCAGGTATAGCATGAGGTCGCTCTTATTGACCACACCTCTACCGG"
# This background sequence includes only the full mutated MUD1 gene
background_seq_2 = "GAAACAAAAAAAAAAAAAGTGGAAGTCAGGGTGTTGGTGTAAAGTCAATGTGACTGCGTTCCACTCCCCACAATAAATTACAAATATCATCACACCCGCTGATCTTACTAGGTGATCAACTGGGCCTTCTTGAAGTTTCTATTTCAAGATCATCAAAGCTGACTAACCAAGCATTTTTAACGTTTGTCACTCAAGAAGAAGCAGACCGGTTTCTAGAAAAATACACGACAACAGCATTAAAAGTTCAAGGTCGTAAAGTGAGACTGGGAAAGACTCGAACAAATTCGTTATTGGGTCATTCAATAGAAATCAAAAAAAAAAAAGGTAATTACGAAACGTACAACCTTGATATAAAGAAGGTGCTTAAAGCAAGGAAACTTAAACGCAAGTTACGTAGTGGTGATATTTGCGCCACAAAGTTCAGGCTTAAAAGGCAAATAAGACGCGTGAAGCATAAGCTGAGATCAAGAAAGGTAGAGGAGGCTGAGAGTGATAGGATTGTAAAAGAATTTGAAACCCGTAGATTCGAAAACATCAAGTCTCAACCAGAAAATCTAAAACAATCGCAGAAACCTCTTAAGCGGGCTAAAGTGTCCAATACAAAGGAAAATACACCGAACAAGGTCCTTCTTATACAAAATTTGCCTAGCGGCACTACCGAGCAATTATCCTCGCAAATACTTGGCAAGGAGGTTTTAGTTGAAATCAGTTTAGTTAGCGTTCGTTACCTAGCTTTCGTGGAATACGAGACCGTTGCTGTTGCTACGAAAATCAAGAATCAGTTAGGCTCCACTTACAAGCTACAAAACAATAACGTTACCATAGGATTTGCTAAGTAGAATATCCTTTGCGGAAGTATACCTCGAGTAAAGAAATTCACAGATAAATTTGAATAACGTTCTCCATTATTAATAAATATTTATTTACAAGCTTAGAAAATAAGGTGCTCTTTATTTACC"

# Indices 
idx_seqs = {1: "AACATCTA", 2: "TGTTGGGA", 3: "AAGCCATG", 4: "GCTAAAGA"}

# Sequence design: Each cur_seq includes: 
# TACGTTAAAC # Left flanking region: remove
# AATACGAGGCACTTACTCCG # Forward primer
# CAAAATATCTGC # Barcode
# TCAGC # Linker
# GTAAGTATATACCTTGTAATTTACGTTTCCTTAAATCTACTAACTGTAGTTGTTTTCATTATTCTATACTAAG # Intron
# TGACCAATAAAAACGGACTGTACTTTCAAAATTTACCCAGTAGGCCAGCAAATAAAGAAAATTATACCAGATTACTTCTG
# CCTGGAGTTCGCTATTCCTA # Reverse primer
# TTGTAGTTTT # Right flanking region: remove
def get_full_rna_seq(cur_seq, background_seq, index_seq,
	insertion_site='TCAATGTGACTGCGTTCCAC', forward_primer='AATACGAGGCACTTACTCCG', reverse_primer='CCTGGAGTTCGCTATTCCTA'):
	insertion_idx = background_seq.index(insertion_site)
	background_part1 = background_seq[0:insertion_idx]
	background_part2 = background_seq[insertion_idx:]
	
	cur_seq_start = cur_seq.index(forward_primer)
	cur_seq_end = cur_seq.index(reverse_primer) + len(reverse_primer)
	cur_seq_segment = cur_seq[cur_seq_start:cur_seq_end]

	full_rna = background_part1 + index_seq	+ cur_seq_segment + background_part2

	return full_rna

def get_fivess_bp_threess_seqs(cur_seq, barcode, splice_site_idxs):
	fivess_idx, bp_idx, threess_idx = splice_site_idxs
	
	left_flank = 'TACGTTAAAC'
	forward_primer = 'AATACGAGGCACTTACTCCG'
	offset = len(left_flank) + len(forward_primer) + len(barcode)
	if VERBOSE: 
		print(offset)
	
	fivess_idx = offset + fivess_idx - 1
	bp_idx = offset + bp_idx - 1
	threess_idx = offset + threess_idx - 1
	
	fivess_seq = cur_seq[fivess_idx:(fivess_idx+20)]
	bp_seq = cur_seq[bp_idx:(bp_idx+20)]
	threess_seq = cur_seq[threess_idx:(threess_idx+20)]
	if VERBOSE:
		print((barcode, fivess_seq, bp_seq, threess_seq))

	return (fivess_seq, bp_seq, threess_seq)

f = open(data_filename, 'w')
 
library_types = {}

lengths = []

df = pd.read_csv(csv_filename)


f.write("type,barcode,splicing_eff,full_seq,fivess_idx,bp_idx,threess_idx\n")

max_intron_len = -1
for idx, row in df.iterrows():
	splicing_effs = {1: row['splicing_eff_idx1'], 2: row['splicing_eff_idx2'], \
		3: row['splicing_eff_idx3'], 4: row['splicing_eff_idx4']}
	cur_seq = row['seq']
	barcode = row['BC']
	library_type = row['type']

	if library_type not in library_types.keys():
		library_types[library_type] = 1
	else: 
		library_types[library_type] += 1

	splice_site_idxs = (int(row['SS5_inds']), int(row['branch_inds']), \
		int(row['SS3_inds']))
	fivess_seq, bp_seq, threess_seq = \
		get_fivess_bp_threess_seqs(cur_seq, barcode, splice_site_idxs)

	for ii in range(1, 5):
		idx_seq = idx_seqs[ii]
		splicing_eff = float(splicing_effs[ii])
		full_seq = get_full_rna_seq(cur_seq, background_seq_1, idx_seq)
		fivess_idx = full_seq.index(fivess_seq)
		bp_idx = full_seq.index(bp_seq)
		threess_idx = full_seq.index(threess_seq)
		if threess_idx - fivess_idx > max_intron_len:
			max_intron_len = threess_idx - fivess_idx
		lengths += [len(full_seq)]
		f.write("%s,%s,%f,%s,%d,%d,%d\n" % \
			(library_type, barcode, splicing_eff, full_seq, \
				fivess_idx, bp_idx, threess_idx))

total = 0
for key, curval in library_types.items():
	print("Library type: %s, Number of items: %d" % (key, curval))
	total += curval
print("Total number of items: %d" % total)

print("Max intron length: %d" % max_intron_len)
# Library types: 
# Library type: endogenous, Number of items: 187
# Library type: endogenous - mutated sites, Number of items: 1306
# Library type: endogenous ss control, Number of items: 450
# Library type: orthologous, Number of items: 961
# Library type: orthologous ss control, Number of items: 988
# Library type: synthetic, Number of items: 4713
# Library type: synthetic alternative background, Number of items: 1377
# Library type: synthetic alternative background control, Number of items: 717
# Library type: synthetic control, Number of items: 618
# Library type: synthetic hairpin, Number of items: 898
# Library type: synthetic hairpin - control, Number of items: 446
# Library type: synthetic mutated, Number of items: 3607

print(np.mean(np.array(lengths)))
print(np.min(np.array(lengths)))
print(np.max(np.array(lengths)))
f.close()

