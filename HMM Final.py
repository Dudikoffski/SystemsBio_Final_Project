# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 18:50:18 2025
Project Final, HMM
@author: Johannes
"""
#%% [0] Dependencies
from datetime import datetime
from hmmlearn import hmm
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from hmmlearn.hmm import CategoricalHMM
import gzip
from Bio import SeqIO
from itertools import groupby
import matplotlib.pyplot as plt
import os
#%% [1] Load Sequences
def load_fasta(file_path):   
    with gzip.open(file_path, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            char_to_int = {'A':0, 'C':1, 'G':2, 'T':3, 'N':4} #Encode sequences
            seq = np.array([
                char_to_int[c] for c in str(record.seq).upper() 
                if c in char_to_int 
#Parse through sequence, encoding every base match found in char_to_int. Needed for HMMlearn
            ], dtype=np.int8).reshape(-1, 1) 
#8bit may be unneccesary, AI recomendation for faster processing. Reshape to fit HMM requirements
            return seq    
    return None  

def compute_state_array(x, window_size=200, cg_threshold=0.5, cpg_oe_threshold=0.6):
    x_squeezed = x.squeeze() #Imported data in nx1, need 1xn
    windows = sliding_window_view(x_squeezed, window_shape=window_size)
    C_counts = np.sum(windows == 1, axis=1)
    G_counts = np.sum(windows == 2, axis=1)
    CG_content = (C_counts + G_counts) / window_size
    #Same as CG filter in HMM post processing
    CpG_observed = np.sum((windows[:, :-1] == 1) & (windows[:, 1:] == 2), axis=1)
    CpG_expected = (C_counts * G_counts) / window_size
    valid = (C_counts > 0) & (G_counts > 0)
    CpG_OE = np.zeros_like(CpG_expected)
    CpG_OE[valid] = CpG_observed[valid] / CpG_expected[valid]
    #Same as O/E post processing
    qualifying_windows = (CG_content > cg_threshold) & (CpG_OE > cpg_oe_threshold)
    #Filtering windows with insufficient CG or O/E content 
    start_indices = np.where(qualifying_windows)[0]
    n = x_squeezed.size
    delta = np.zeros(n + 1, dtype=int)
    delta[start_indices] += 1
    delta[start_indices + window_size] -= 1
    mask = (np.cumsum(delta)[:n] > 0)
    #Decides CGI state based on at least one window recognizing a nucleotide as part of one
    return mask.astype(int).reshape(-1, 1)
#Sliding window for comparison to HMM. 

path_Chr22 = 'C:/Users/Johan/Documents/Systembiologie/Sequences/chr22.fa.gz' 
sequence_Chr22 = load_fasta(path_Chr22)
path_Chr1 = 'C:/Users/Johan/Documents/Systembiologie/Sequences/chr1.fa.gz' 
sequence_Chr1 = load_fasta(path_Chr1)
#%% [2] Define Model
HMM = CategoricalHMM(n_components = 3,  
#3 components: nonCGI, CGI, poly-N. Technically not needed, since initialized arrays induce 3-factor already
                     algorithm = 'viterbi', 
#Only used for decoding. Training with fit function automatically uses Baum-Welch
                     init_params = '',
#Stops reinitialization of parameters upon training start with random parameters
                     n_iter = 10) 
# Higher iteration counts start eating away at one of the states very existence...
#%% [3] Set Starting Parameters
HMM.startprob_ = np.array([.1,.1,.8]) 
#Chromosomes always start and end with poly-N, so 0,0,1 would also work 
HMM.transmat_ = np.array([[.95,.025,.025],
                          [.075,.90,.025],
                          [.025,.025,.95]])
#All states, by definition, are long and uninterrupted regions. Therefore transitions between regions should be rare.
HMM.emissionprob_ = np.array([[.29,.19,.19,.29,.04],
                              [.04,.45,.45,.04,.02],
                              [.01,.01,.01,.01,.96]])
#CG decay in non-CGI, high CG count in CGI, and only N in poly-N inform our guesses for emissions.
#%% [4] Model Training
HMM.fit(sequence_Chr1)
#%% [4.5] Alternative (training time intensive)Load Pretrained Matrices
# HMM.startprob_ = np.loadtxt("C:/Users/Johan/Documents/Systembiologie/Sequences/start_matrix_20250327_180114.csv", delimiter=",").flatten()
# HMM.transmat_ = np.loadtxt("C:/Users/Johan/Documents/Systembiologie/Sequences/transition_matrix_20250327_180114.csv", delimiter=",")
# HMM.emissionprob_ = np.loadtxt("C:/Users/Johan/Documents/Systembiologie/Sequences/emission_matrix_20250327_180114.csv", delimiter=",")
#Alternitavely, a specific seed could be forced prior to training to keep results comparable.
#%% [5] Decode Sequence
log_prob, decoded_states = HMM.decode(sequence_Chr22)
slide_decode = compute_state_array(sequence_Chr22)
#%% [6] Filter& Unify CGI Regions
Indices = np.where(decoded_states == 1)[0]
regions = []
Indices_1 = np.where(slide_decode == 1)[0]
regions_1= []

for key, group in groupby(enumerate(Indices_1), key=lambda x: x[0] - x[1]):
    group_list = list(group)
    start = group_list[0][1] 
    end = group_list[-1][1]   
    regions_1.append((start, end))

for key, group in groupby(enumerate(Indices), key=lambda x: x[0] - x[1]): 
    group_list = list(group)
    start = group_list[0][1] 
    end = group_list[-1][1]   
    regions.append((start, end))
#Forms difference of neighboring entries in Indices. Groups are made up of consecutive equal differences, denoting no gaps in their CGI decoding.


min_length = 200
filtered_regions_0 = [(start, end) for (start, end) in regions
                    if (end - start) >= min_length]
#Filters out groups not meeting the length criteria of CGIs

min_CG = 0.50
filtered_regions_1 = [(start,end) for (start, end) in filtered_regions_0
                    if  (np.sum( (sequence_Chr22[start:end+1] == 1) 
                               | (sequence_Chr22[start:end+1] == 2) )
                    / (end - start + 1) >= min_CG)]
#Adds together all counts of C and G and checks against CG content ratio

min_OE = 0.6
filtered_regions_2 = [(start, end)
    for (start, end) in filtered_regions_1
    if  (lambda start, end: (
        (O := np.sum((sequence_Chr22[start:end] == 1) & 
                     (sequence_Chr22[start+1:end+1] == 2))) >0 and
#Checks and counts instances where a C is followed by a G (CpG)
        (C := np.sum(sequence_Chr22[start:end] == 1)) >0 and
        (G := np.sum(sequence_Chr22[start:end] == 2)) >0 and
#Counts Cs and Gs
        ((O * (end - start)) / (C * G) >= min_OE)  
    ))(start, end)]
#Checks against O/E ratio and filters out lower O/E groups
#%% [7] Histogram of Island Lengths
lengths = [end - start + 1 for (start, end) in merged_regions]
BED_regions = np.genfromtxt(
    'C:/Users/Johan/Documents/Systembiologie/Sequences/Hg38_chr22_text.txt',
    delimiter="\t",
    skip_header=1,  
    usecols=(1, 2),  
    dtype=int)                                
lengths_BED = [end - start + 1 for (start, end) in BED_regions] 
#Calculate CGI lengths for histogram

bins = np.linspace(200, max(max(lengths_BED), max(lengths)), 50)             
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#Calculate bin sizes ranging from 200 (min theoretical) to largest CGI in HMM or BED data

ax1.hist(lengths_BED, bins=bins, color='navy', edgecolor='black', label='BED-File lengths')
ax1.set_title('Sequence Lengths (BED)')
ax1.set_xlabel('Length')
ax1.set_ylabel('Frequency')

ax2.hist(lengths, bins=bins, color='red', edgecolor='black', label='lengths')
ax2.set_title('Sequence Lengths (HMM)')
ax2.set_xlabel('Length')

plt.tight_layout()
plt.show()
#%% [8] Export Regions for Bedtools Analysis
output_dir = "C:/Users/Johan/Documents/Systembiologie/Sequences"
output_bed = os.path.join(output_dir, "Chr22.bed")
output_bed_1= os.path.join(output_dir, "Slide_22.bed")
                               
with open(output_bed, 'w') as f:
    for start, end in filtered_regions_2:
        f.write(f"chr22\t{start}\t{end}\n")
        
with open(output_bed_1, 'w') as f:
    for start, end in regions:
        f.write(f"chr22\t{start}\t{end}\n")
#Saves CGI regions (start, end) as BED files
#%% [9] Bedtools commands (Linux Environment)
#The following commands were executed in a Linux environment using the bedtools suite
#Results are copied directly, but could theoretically be saved and imported
"cd /mnt/c/Users/Johan/Documents/Systembiologie/Sequences"
"bedtools jaccard -a Hg38_chr22.bed -b Chr22.bed"
JS = 0.595122
#Jaccard compares the length of the sequence overlap to the combined "shadow" of both sequences

"bedtools intersect -a Hg38_chr22.bed -b Chr22.bed -wao \
| awk 'BEGIN {TP=0} {TP+=$NF} END {print ''TP =', TP}'"
TP = 391761 
#Total length of true positive sequences

"bedtools subtract -a Hg38_chr22.bed -b Chr22.bed \
| bedtools merge \| awk 'BEGIN {FN=0} {FN+=$3-$2} END {print 'FN =', FN}'"
FN= 199171
#Total length of false negative sequences

"bedtools subtract -a Chr22.bed -b Hg38_chr22.bed \
| bedtools merge \| awk 'BEGIN {FP=0} {FP+=$3-$2} END {print 'FP =', FP}'"
FP= 67355
#Total length of false positive sequences

TN = len(sequence_Chr22)-(TP+FN+FP) #True negative

"cd /mnt/c/Users/Johan/Documents/Systembiologie/Sequences"
"bedtools jaccard -a Hg38_chr22.bed -b Slide_22.bed"
JS_1 = 0.15516

"bedtools intersect -a Hg38_chr22.bed -b Slide_22.bed -wao \
| awk 'BEGIN {TP=0} {TP+=$NF} END {print ''TP =', TP}'"
TP_1 = 580608

"bedtools subtract -a Hg38_chr22.bed -b Slide_22.bed \
| bedtools merge \ | awk 'BEGIN {FN=0} {FN+=$3-$2} END {print 'FN =', FN}'"
FN_1 = 10324

"bedtools subtract -a Slide_22.bed -b Hg38_chr22.bed \
| bedtools merge \ | awk 'BEGIN {FP=0} {FP+=$3-$2} END {print 'FP =', FP}'"
FP_1 = 3151073

TN_1 = len(sequence_Chr22)-(TP_1+FN_1+FP_1)
#%% [10] Calculating Fit Ratings
#Calculate Sensitivity, Specificity, Precission and F-Score
Sens = TP / (TP + FN)
Spec = TN / (TN + FP)
Prec = TP / (TP + FP)
F_score = 2 * (Prec * Sens) / (Prec + Sens)
print ('Sensitivity  HMM =',Sens)
print ('Specificity  HMM =',Spec)
print ('F-Score      HMM =',F_score)
print('Jaccard-Score HMM =',JS)

Sens_1 = TP_1 / (TP_1 + FN_1)
Spec_1 = TN_1 / (TN_1 + FP_1)
Prec_1 = TP_1 / (TP_1 + FP_1)
F_score_1 = 2 * (Prec_1 * Sens_1) / (Prec_1 + Sens_1)
print ('Sensitivity  Slide =',Sens_1)
print ('Specificity  Slide =',Spec_1)
print ('F-Score      Slide =',F_score_1)
print('Jaccard-Score Slide =',JS_1)
