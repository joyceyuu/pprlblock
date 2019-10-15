from numpy import *
import matplotlib.pyplot as pl
import pandas as pd
import numpy as np
import gzip
import hashlib
import math
import random
import string
import sys
import os
import time
import bisect
from itertools import tee
import re
import pandas as pd
from collections import defaultdict
import multiprocessing
### self dependency
from simmeasure import DiceSim, BloomFilterSim, editdist
from pprlindex import PPRLIndex
from pprlpsig import PPRLIndexPSignature

data_sets_pairs = [
  # ['./datasets/4611_50_overlap_no_mod_alice.csv',
  #  './datasets/4611_50_overlap_no_mod_bob.csv'],
  # #
  # ['./datasets/46116_50_overlap_no_mod_alice.csv',
  #  './datasets/46116_50_overlap_no_mod_bob.csv'],

  ['./datasets/461167_50_overlap_no_mod_alice.csv',
   './datasets/461167_50_overlap_no_mod_bob.csv'],
  # #
  # ['./datasets/4611676_50_overlap_no_mod_alice.csv',
  #  './datasets/4611676_50_overlap_no_mod_bob.csv'],

]

oz_attr_sel_list = [1,2]
attr_bf_sample_list = [60,40]  # Sample 50% of bits from attribute BF
dice_sim = DiceSim()
bf_sim =   BloomFilterSim()


assess_results = []
# drop_ratio = np.linspace(0.001, 0.6, 5)
drop_ratio = np.linspace(0.001, 0.05, 20)
# drop_ratio = [0.01]

def psig_block_tune(psig, dr):
    """Fit a psig with given dropout ratio."""
    psig.common_bloom_filter([1, 2])
    start_time = time.time()
    psig.drop_toofrequent_index(len(psig.rec_dict_alice) * dr)
    a_min_blk,a_med_blk,a_max_blk,a_avg_blk,a_std_dev = psig.build_index_alice()
    b_min_blk,b_med_blk,b_max_blk,b_avg_blk,b_std_dev = psig.build_index_bob()
    dbo_time = time.time() - start_time

    start_time = time.time()
    num_blocks = psig.generate_blocks()
    lu_time = time.time() - start_time
    print('P-Signature Finish generating blocks!')

    # find number of candidate pairs
    num_cand_rec_pairs = 0
    for bkkey, bk in psig.block_dict.items():
        alice_rec_id_list, bob_rec_id_list = bk
        n1, n2 = len(alice_rec_id_list), len(bob_rec_id_list)
        num_cand_rec_pairs += n1 * n2
    return psig, num_cand_rec_pairs


alice_data_set, bob_data_set = data_sets_pairs[0]

# num of reference values R = N/k
alice_dataset_str = alice_data_set.split('/')[-1]
alice_num_recs = int(alice_dataset_str.split('_')[0])
#alice_num_recs = 481315  # NC

bob_dataset_str = bob_data_set.split('/')[-1]
bob_num_recs = int(bob_dataset_str.split('_')[0])
#bob_num_recs = 480701   # NC

num_recs = max(alice_num_recs,bob_num_recs)

# build a P-Sig instance
psig = PPRLIndexPSignature(num_hash_funct=20, bf_len=1024)
psig.load_database_alice(alice_data_set, header_line=True,
                       rec_id_col=0, ent_id_col=0)
psig.load_database_bob(bob_data_set, header_line=True,
                       rec_id_col=0, ent_id_col=0)

cand_pairs = []
for dr in drop_ratio:
    print(dr)
    psig, num_cand_rec_pairs = psig_block_tune(psig, dr)
    cand_pairs.append(num_cand_rec_pairs)

snn_sim = 138530971
kasn = 116958866
hlsh_clust = 272999606

# import IPython; IPython.embed()
pl.figure(figsize=(12, 12))
pl.plot(drop_ratio, cand_pairs, label='psig')
pl.axhline(y=snn_sim, label='snn_sim', color='r')
pl.axhline(y=kasn, label='kasn', color='g')
pl.axhline(y=hlsh_clust, label='hlsh_clust', color='m')
pl.xlabel('Dropout Ratio')
pl.legend()
pl.yscale('log')
pl.ylabel('Number of Candidate Pairs')
pl.title('Dropout Ratio versus Number of Candidate Pairs')
pl.grid()
pl.savefig('ratio_vs_candpairs.pdf')
