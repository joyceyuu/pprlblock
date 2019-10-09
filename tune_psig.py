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
### self dependency
from simmeasure import DiceSim, BloomFilterSim, editdist
from pprlindex import PPRLIndex
from pprlpsig import PPRLIndexPSignature

data_sets_pairs = [
  # ['./datasets/4611_50_overlap_no_mod_alice.csv',
  #  './datasets/4611_50_overlap_no_mod_bob.csv'],
  #
  ['./datasets/46116_50_overlap_no_mod_alice.csv',
   './datasets/46116_50_overlap_no_mod_bob.csv'],

  # ['./datasets/461167_50_overlap_no_mod_alice.csv',
  #  './datasets/461167_50_overlap_no_mod_bob.csv'],
  #
  # ['./datasets/4611676_50_overlap_no_mod_alice.csv',
  #  './datasets/4611676_50_overlap_no_mod_bob.csv'],

]

oz_attr_sel_list = [1,2]
attr_bf_sample_list = [60,40]  # Sample 50% of bits from attribute BF
dice_sim = DiceSim()
bf_sim =   BloomFilterSim()


assess_results = []
# drop_ratio = np.linspace(0.001, 0.6, 5)
drop_ratio = [0.03, 0.04, 0.05]


for (alice_data_set, bob_data_set) in data_sets_pairs:
    K=100
    oz_small_alice_file_name = alice_data_set
    oz_small_bob_file_name   = bob_data_set
    # num of reference values R = N/k
    alice_dataset_str = alice_data_set.split('/')[-1]
    alice_num_recs = int(alice_dataset_str.split('_')[0])
    #alice_num_recs = 481315  # NC

    bob_dataset_str = bob_data_set.split('/')[-1]
    bob_num_recs = int(bob_dataset_str.split('_')[0])
    #bob_num_recs = 480701   # NC

    num_recs = max(alice_num_recs,bob_num_recs)
    num_ref_val = num_recs/K

    for dr in drop_ratio:
        psig = PPRLIndexPSignature(num_hash_funct=20, bf_len=1024)
        psig.load_database_alice(oz_small_alice_file_name, header_line=True,
                               rec_id_col=0, ent_id_col=0)
        psig.load_database_bob(oz_small_bob_file_name, header_line=True,
                               rec_id_col=0, ent_id_col=0)
        start_time = time.time()
        psig.common_bloom_filter([1, 2])
        psig.drop_toofrequent_index(len(psig.rec_dict_alice) * dr)
        a_min_blk,a_med_blk,a_max_blk,a_avg_blk,a_std_dev = psig.build_index_alice()
        b_min_blk,b_med_blk,b_max_blk,b_avg_blk,b_std_dev = psig.build_index_bob()
        dbo_time = time.time() - start_time

        start_time = time.time()
        num_blocks = psig.generate_blocks()
        lu_time = time.time() - start_time
        rr, pc, pq, num_cand_rec_pairs = psig.assess_blocks()
        assess_results.append(['psig', dr,
            alice_num_recs, bob_num_recs, num_ref_val, K,
            dbo_time, lu_time, rr, pc, pq,
            a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
            b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
            num_blocks, num_cand_rec_pairs
        ])

# dataframe that summarize all methods
df = pd.DataFrame(data=assess_results)
df.columns = ['Method', 'drop_ratio',
    'alice_num_recs', 'bob_num_recs', 'num_ref_val', 'K',
    'dbo_time', 'lu_time', 'rr', 'pc', 'pq',
    'a_min_blk', 'a_med_blk', 'a_max_blk', 'a_avg_blk', 'a_std_dev',
    'b_min_blk', 'b_med_blk', 'b_max_blk', 'b_avg_blk', 'b_std_dev',
    'num_blocks', 'num_cand_rec_pairs']
print()
print(df)
df.to_csv('psig.csv', index=False)

# import IPython; IPython.embed()
res = pd.read_csv('psig.csv')
x = res['drop_ratio'].values
rr = res['rr'].values
pc = res['pc'].values
pl.figure(figsize=(8, 6))
pl.plot(x, rr, label='reduction ratio')
pl.plot(x, pc, label='pair completeness')
pl.title('Reduction Ratio and Pair Completeness versus Drop Ratio (n={})'
         .format(alice_num_recs))
pl.grid()
pl.legend()
pl.show()
