"""
Make predictions for the splicing efficiency values using a baseline model: 
Predicts 0 if the sequence is not similar to known splice sites in wildtype yeast
If the sequence is similar, predict the average splicing efficiency value from the training dataset
for non-zero examples

Evaluates performance of the baseline model on the train and dev set using the loss function

Similarity to known splice sites is governed by: 

1. Whether sequence features (5' splice site, branch point, 3' splice site) fall within 
the 95% of the wildtype yeast spliced sequences' position-weight matrices (PWMs)

2. Does the length distribution of the full intron and distance between the 
(5' splice site, branch point, 3' splice site) fall within the 95% of wildtype yeast introns?

Example usage: python baseline_model.py ../data/wildtype_introns.csv ../data/train_dev_test/train.csv ../data/train_dev_test/dev.csv
"""

import sys
import math
import random
import numpy as np 
import pandas as pd
from attr import attrs,attrib

from utils import compute_loss

FIVESS_LEN = 6 # Length of 5'SS motif from the wildtype yeast intron dataset
BP_LEN = 8 # Length of BP motif from the wildtype yeast intron dataset
THREESS_LEN = 3 # Length of 3'SS motif from the wildtype yeast intron dataset
BP_SHIFT = 6 # How far into the BP motif is the branchpoint?
THREEPRIME_SHIFT = 2 # How far into the 3'SS motif is the end of the intron?
EPSILON = 0.001 # Error allowed when comparing PWM match scores

PERCENTILE=0.95

nt_idxs = {"A": 0, "T": 1, "C": 2, "G": 3, "X": 1}


wildtype_filename = sys.argv[1]
wildtype_df = pd.read_csv(wildtype_filename)

train_filename = sys.argv[2]
train_df = pd.read_csv(train_filename)

dev_filename = sys.argv[3]
dev_df = pd.read_csv(dev_filename)


@attrs
class IntronPWMs:
    fiveprime_pwm = attrib()
    threeprime_pwm = attrib()
    bp_pwm = attrib()
    fiveprime_cutoff = attrib()
    threeprime_cutoff = attrib()
    bp_cutoff = attrib()

@attrs
class LengthCutoffs:
    min_length = attrib()
    max_length = attrib()
    min_bp_end = attrib()
    max_bp_end = attrib()
    min_start_bp = attrib()
    max_start_bp = attrib()

@attrs
class Candidate:
    fiveprime_seq = attrib()
    bp_seq = attrib()
    threeprime_seq = attrib()
    fiveprime_idx = attrib()
    bp_idx = attrib()
    threeprime_idx = attrib()

# Gets the PWM from a set of sequences
# Requirements: All sequences are the same length, 
# consist of ATCG, and includes at least 1 sequence.
def get_pwm(seqs):
    counts = np.array([1] * len(seqs[0]) * 4) 
    counts.resize((4, len(seqs[0])))

    for seq in seqs:
        for ii, curchar in enumerate(seq):
            counts[nt_idxs[curchar], ii] += 1

    counts = counts/counts.sum(axis=0, keepdims=True)
    return np.log(counts/0.25)

def get_score(pwm, seq):
    align_score = 0
    for jj, curchar in enumerate(seq):
        align_score += pwm[nt_idxs[curchar.upper()], jj]
    return align_score

# Get sorted scores for all sequences based on the given pwm
def get_score_list(pwm, seqs):
    score_list = np.zeros(len(seqs))
    for ii, seq in enumerate(seqs): 
        align_score = 0
        for jj, curchar in enumerate(seq):
            align_score += pwm[nt_idxs[curchar], jj]
        score_list[ii] = align_score
    score_list = np.sort(score_list)
    return score_list

# Get the pwm score at a particular percentile for the list of sequences
def get_score_percentile(pwm, seqs, perc):
    score_list = get_score_list(pwm, seqs)
    return score_list[int(len(score_list) * perc)]

# Get the pwm score at each percentile for the list of sequences
def get_score_percentiles(pwm, seqs):
    score_list = get_score_list(pwm, seqs)
    percs = [0, 0.01, 0.25, 0.5, 0.75, 0.9, 0.999]
    percentile_list = []
    for perc in percs:
        percentile_list += [score_list[int(len(score_list) * perc)]]
    return percentile_list

# Get the 5'SS, BP, and 3'SS PWM's and percentile cutoffs given a 
# df that has the 5'SS, BP, and 3'SS sequences
def get_pwms(df, perc=0.1, verbose=False):
    fiveprime_seqs = list(df["5'SS"])
    bp_seqs = list(df["BP"])
    threeprime_seqs = list(df["3'SS"])

    fiveprime_pwm = get_pwm(fiveprime_seqs)
    fiveprime_cutoff = get_score_percentile(fiveprime_pwm, fiveprime_seqs, perc)
    if verbose:
        score_perc = get_score_percentiles(fiveprime_pwm, fiveprime_seqs)
        print("Getting 5'SS PWM")
        print(fiveprime_pwm)
        print(score_perc)
    print("5'SS PWM cutoff: %f" % fiveprime_cutoff)
    
    bp_pwm = get_pwm(bp_seqs)
    bp_cutoff = get_score_percentile(bp_pwm, bp_seqs, perc)
    if verbose:
        score_perc = get_score_percentiles(bp_pwm, bp_seqs)
        print("Getting BP PWM")
        print(bp_pwm)
        print(score_perc)
    print("BP PWM cutoff: %f" % bp_cutoff)
    
    threeprime_pwm = get_pwm(threeprime_seqs)
    threeprime_cutoff = get_score_percentile(threeprime_pwm, threeprime_seqs, perc)
    if verbose:
        score_perc = get_score_percentiles(threeprime_pwm, threeprime_seqs)
        print("Getting 3'SS PWM")
        print(threeprime_pwm)
        print(score_perc)
    print("3'SS PWM cutoff: %f" % threeprime_cutoff)
    
    return IntronPWMs(fiveprime_pwm=fiveprime_pwm, threeprime_pwm=threeprime_pwm, bp_pwm=bp_pwm, \
        fiveprime_cutoff=fiveprime_cutoff, threeprime_cutoff=threeprime_cutoff, bp_cutoff=bp_cutoff)

def get_perc_values(values, perc):
    values = np.sort(np.array(values).astype(int))
    min_value = values[int(len(values) * perc)]
    max_value = values[int(len(values) * (1 - perc))]
    return (min_value, max_value)

# Get percentile cutoffs for lengths between start, BP, and end
def get_length_cutoffs(df, perc=0.1):
    (min_length, max_length) = get_perc_values(list(df["Length (bp)"]), perc)
    (min_bp_end, max_bp_end) = get_perc_values(list(df["BP to 3'SS (bp)"]), perc)
    (min_start_bp, max_start_bp) = get_perc_values(list(df["5'SS to BP (bp)"]), perc)
    
    return LengthCutoffs(min_length=min_length, max_length=max_length, \
        min_bp_end=min_bp_end, max_bp_end=max_bp_end, min_start_bp=min_start_bp, max_start_bp=max_start_bp)

# Predicts whether the Candidate splices using the cutoffs computed
# in pwms and len_cutoffs
# Returns 1 if Candidate splices, 0 otherwise
def check_splicing(candidate, pwms, len_cutoffs):
    fiveprime_score = get_score(pwms.fiveprime_pwm, candidate.fiveprime_seq)
    threeprime_score = get_score(pwms.threeprime_pwm, candidate.threeprime_seq) 
    bp_score = get_score(pwms.bp_pwm, candidate.bp_seq)

    intron_len = candidate.threeprime_idx + THREESS_LEN - candidate.fiveprime_idx
    fiveprime_bp_len = candidate.bp_idx + BP_SHIFT - candidate.fiveprime_idx + 1
    bp_threeprime_len = candidate.threeprime_idx + THREESS_LEN - candidate.bp_idx - BP_SHIFT

    if fiveprime_score + EPSILON < pwms.fiveprime_cutoff or \
        threeprime_score + EPSILON < pwms.threeprime_cutoff or \
        bp_score + EPSILON < pwms.bp_cutoff: 
        return 0

    if intron_len > len_cutoffs.max_length or \
        intron_len < len_cutoffs.min_length: 
        return 0
    #if fiveprime_bp_len > len_cutoffs.max_start_bp or \
    #    fiveprime_bp_len < len_cutoffs.min_start_bp: 
    #    return 0
    if bp_threeprime_len > len_cutoffs.max_bp_end or \
        bp_threeprime_len < len_cutoffs.min_bp_end:
        return 0

    return 1

# Assemble Candidate objects from data file
def get_candidates(candidate_df):
    candidates = []
    splicing_effs = []


    for index, row in candidate_df.iterrows():
        fiveprime_idx = row['fivess_idx']
        bp_idx = row['bp_idx']
        threeprime_idx = row['threess_idx']
        full_seq = row['full_seq']
     
        fiveprime_seq = full_seq[fiveprime_idx:(fiveprime_idx + FIVESS_LEN)]
        bp_seq = full_seq[bp_idx:(bp_idx + BP_LEN)]
        threeprime_seq = full_seq[threeprime_idx:(threeprime_idx + THREESS_LEN)]
        
        candidates += [Candidate(fiveprime_seq=fiveprime_seq, bp_seq=bp_seq, threeprime_seq=threeprime_seq, \
            fiveprime_idx=fiveprime_idx, bp_idx=bp_idx, threeprime_idx=threeprime_idx)]
        splicing_effs += [float(row['splicing_eff'])]

    return candidates, splicing_effs

def get_nonzero_mean_splicing_eff(df):
    splicing_effs = df['splicing_eff'].astype(float)
    splicing_effs = splicing_effs[splicing_effs > EPSILON]
    return np.mean(splicing_effs)

def get_baseline_loss(candidate_df, pwms, len_cutoffs, mean_val, verbose=False):
    candidates, splicing_effs = get_candidates(candidate_df)
    splicing_effs = np.array(splicing_effs)

    pred = np.zeros(len(candidates))
    for ii, candidate in enumerate(candidates):
        pred[ii] = check_splicing(candidate, pwms, len_cutoffs)

    pred *= mean_val # Predict the mean splicing efficiency value for non-zero values

    if verbose:
        print("Average prediction value: %f" % np.mean(pred))
        print("Average true value: %f" % np.mean(splicing_effs))

    return compute_loss(pred, splicing_effs)

pwms = get_pwms(wildtype_df, perc=(1-PERCENTILE))
len_cutoffs = get_length_cutoffs(wildtype_df, perc=(1-PERCENTILE))
print(len_cutoffs)

train_nonzero_mean = get_nonzero_mean_splicing_eff(train_df)
dev_loss = get_baseline_loss(dev_df, pwms, len_cutoffs, train_nonzero_mean)
print("Loss on dev set: %f" % dev_loss)
