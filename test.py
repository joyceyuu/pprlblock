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
from itertools import tee, product
import re
import pandas as pd
from collections import defaultdict
### self dependency
from simmeasure import DiceSim, BloomFilterSim, editdist
from pprlindex import PPRLIndex
from pprlpsig import PPRLIndexPSignature
import networkx as nx
from tqdm import tqdm

data_sets_pairs = [
  # ['./datasets/4611_50_overlap_no_mod_alice.csv',
  #  './datasets/4611_50_overlap_no_mod_bob.csv'],
  #
  ['./datasets/46116_50_overlap_no_mod_alice.csv',
   './datasets/46116_50_overlap_no_mod_bob.csv'],

  # ['./datasets/461167_50_overlap_no_mod_alice.csv',
  #  './datasets/461167_50_overlap_no_mod_bob.csv'],
  # #
  # ['./datasets/4611676_50_overlap_no_mod_alice.csv',
  #  './datasets/4611676_50_overlap_no_mod_bob.csv'],

]

oz_attr_sel_list = [1,2]
attr_bf_sample_list = [60,40]  # Sample 50% of bits from attribute BF
dice_sim = DiceSim()
bf_sim =   BloomFilterSim()

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
psig = PPRLIndexPSignature(num_hash_funct=20, bf_len=2048, gram_n=3)
psig.load_database_alice(alice_data_set, header_line=True,
                       rec_id_col=0, ent_id_col=0)
psig.load_database_bob(bob_data_set, header_line=True,
                       rec_id_col=0, ent_id_col=0)
psig.common_bloom_filter([1, 2])

dr = 0.0015
k = 0.00002
start_time = time.time()
n = len(psig.rec_dict_alice)
psig.drop_toofrequent_index(n * dr, max(n * k, 1))
a_min_blk,a_med_blk,a_max_blk,a_avg_blk,a_std_dev = psig.build_index_alice()
b_min_blk,b_med_blk,b_max_blk,b_avg_blk,b_std_dev = psig.build_index_bob()
dbo_time = time.time() - start_time

start_time = time.time()
num_blocks = psig.generate_blocks()
lu_time = time.time() - start_time
print('P-Signature Finish generating blocks!')

# sort blocks from highest to lowest
blocks = psig.block_dict
keys = sorted(blocks, key=lambda k: len(blocks[k][0]) * len(blocks[k][1]),
              reverse=True)
blocks = {k: blocks[k] for k in keys}

alen = [len(x[0]) for x in blocks.values()]
blen = [len(x[1]) for x in blocks.values()]

pl.subplot(1, 2, 1)
pl.hist(alen, bins=20)
pl.subplot(1, 2, 2)
pl.hist(blen, bins=20)
pl.show()

# find number of total candidate pairs
num_rec_alice = len(psig.rec_dict_alice)
num_rec_bob = len(psig.rec_dict_bob)
total_rec = num_rec_alice * num_rec_bob

cand_pairs_time = time.time()
# find number of candidate pairs with blocking
cand_pairs = set()
print('Finding number of candidate pairs...')
for i, (alice_rids, bob_rids) in blocks.items():
    n = len(alice_rids) * len(bob_rids)
    print('Processing block={} number of pairs={:,}'.format(i, n))
    for a, b in tqdm(product(alice_rids, bob_rids)):
        cand_pairs.add((a, b))
print("Total number of record pairs:          {}".format(total_rec))
print("Number of candidate record pairs:      {}".format(len(cand_pairs)))
delta_cand_pairs = time.time() - cand_pairs_time
print('Total time for calculate candidate pairs={}'.format(delta_cand_pairs))

# calculate rr, pc, pq
num_block_true_matches = 0
num_block_false_matches = 0
for a, b in tqdm(cand_pairs):
    if a == b:
        num_block_true_matches += 1
    else:
        num_block_false_matches += 1

rr = 1.0 - float(len(cand_pairs)) / float(total_rec)
alice = psig.rec_dict_alice
bob = psig.rec_dict_bob
num_all_true_matches = len(set(alice.keys()).intersection(set(bob.keys())))
if num_block_true_matches > 0:
    pc = float(num_block_true_matches) / float(num_all_true_matches)
else:
    pc = -1.0

if len(cand_pairs) > 0:
    pq = float(num_block_true_matches) / float(len(cand_pairs))
else:
    pq = -1.0

rr, pc, pq, num_cand_rec_pairs = psig.assess_blocks()
algorithem_name = 'psig'
print('Quality and complexity results for {}:'.format(algorithem_name))
print('  rr = %.4f' % (rr))
print('  pc = %.4f' % (pc))
print('  pq = %.4f' % (pq))
print('  dbo time = %.4f' % (dbo_time))
print('  lu time = %.4f' % (lu_time))
#
#
# K = 100
# num_ref_val = num_recs/K
# assess_results = []
# assess_results.append(['psig',
#     alice_num_recs, bob_num_recs, num_ref_val, K,
#     dbo_time, lu_time, rr, pc, pq,
#     a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
#     b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
#     num_blocks, num_cand_rec_pairs
# ])
#
#
# # dataframe that summarize all methods
# df = pd.DataFrame(data=assess_results)
# df.columns = ['Method',
#     'alice_num_recs', 'bob_num_recs', 'num_ref_val', 'K',
#     'dbo_time', 'lu_time', 'rr', 'pc', 'pq',
#     'a_min_blk', 'a_med_blk', 'a_max_blk', 'a_avg_blk', 'a_std_dev',
#     'b_min_blk', 'b_med_blk', 'b_max_blk', 'b_avg_blk', 'b_std_dev',
#     'num_blocks', 'num_cand_rec_pairs']
# print()
# print(df)
# n = df['alice_num_recs'].unique()[0]
# df.to_csv('psig_n={}.csv'.format(n), index=False)
# G = nx.Graph()
#
# for i, (alice_rids, bob_rids) in blocks.items():
#     print('Processing block={}'.format(i))
#     # alice_rids = set(alice_rids)
#     # bob_rids = set(bob_rids)
#     n1 = len(alice_rids)
#     n2 = len(bob_rids)
#     print('Alice={} Bob={} Total Combination={}'.format(n1, n2, n1 * n2))
#
#     alice_dup = len(set(alice_rids)) < len(alice_rids)
#     bob_dup = len(set(bob_rids)) < len(bob_rids)
#     print('There is duplicates in Alice={}'.format(alice_dup))
#     print('There is duplicates in Bob={}'.format(bob_dup))
#
#     # add nodes
#     G.add_nodes_from(['a_{}'.format(x) for x in alice_rids])
#     G.add_nodes_from(['b_{}'.format(x) for x in bob_rids])
#
#     batch = int(n1 * n2 / 100)
#     # add edges
#     for j, (a, b) in enumerate(product(alice_rids, bob_rids)):
#         if j % batch == 0:
#             progress = j / (n1 * n2)
#             print('Processing edges={}%'.format(round(progress * 100, 2)))
#         G.add_edge(a, b)

# import IPython; IPython.embed()
