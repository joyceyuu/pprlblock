from numpy import *
import numpy
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
from pprlknn import PPRLIndexKAnonymousNearestNeighbourClustering
from pprlknnsorted import PPRLIndexKAnonymousSortedNeighbour
from pprlbloomfilterhlsh import PPRLIndexBloomFilterHLSH
from pprl2partyknnsorted import PPRLIndex2PartyKAnonymousSortedNeighbour
from pprlhclustering import hclustering
from pprlpsig import PPRLIndexPSignature
from vis import draw_risk, draw_riskcompare


MIN_SIM_VAL =       0.8   # 0.9 , 0.8, 0.6 - SNN3P, SNN2P, kNN
#K =                100
W =                 2 #4,7,8
OVERLAP =           0

N_HASH =       30
SET_BIT_PERC = 50
RAND_SEED =    42

HLSH_NUM_BIT = 45
HLSH_NUM_ITER = 40
# oz_file_name = 'datasets/OZ-clean-with-gname.csv'
oz_file_name = 'datasets/OZ-clean-with-gname.csv'
#nc_file_name = 'datasets/ncvoter-temporal.csv'


data_sets_pairs = [
  ['./datasets/46116_50_overlap_no_mod_alice.csv',
   './datasets/46116_50_overlap_no_mod_bob.csv']]
oz_attr_sel_list = [1,2]
#nc_attr_sel_list = [3,5]

attr_bf_sample_list = [60,40]  # Sample 50% of bits from attribute BF
dice_sim = DiceSim()

bf_sim =   BloomFilterSim()
K = 100
for (alice_data_set, bob_data_set) in data_sets_pairs:
    assess_results = []
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

    psig = PPRLIndexPSignature(num_hash_funct=20, bf_len=1024)
    psig.load_database_alice(oz_small_alice_file_name, header_line=True,
                           rec_id_col=0, ent_id_col=0)
    psig.load_database_bob(oz_small_bob_file_name, header_line=True,
                           rec_id_col=0, ent_id_col=0)
    start_time = time.time()
    psig.common_bloom_filter([1, 2])
    psig.drop_toofrequent_index(len(psig.rec_dict_alice) * 0.05)
    a_min_blk,a_med_blk,a_max_blk,a_avg_blk,a_std_dev = psig.build_index_alice()
    b_min_blk,b_med_blk,b_max_blk,b_avg_blk,b_std_dev = psig.build_index_bob()
    dbo_time = time.time() - start_time

    start_time = time.time()
    num_blocks = psig.generate_blocks()
    lu_time = time.time() - start_time
    rr, pc, pq, num_cand_rec_pairs = psig.assess_blocks()

    psigarisk, psigbrisk = psig.disclosure_risk()
    # draw_risk(arisk, brisk, 'P-Sig', alice_num_recs)

    assess_results.append(['p3-sig',
        alice_num_recs, bob_num_recs, num_ref_val, K,
        dbo_time, lu_time, rr, pc, pq,
        a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
        b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
        num_blocks, num_cand_rec_pairs
    ])

    # import IPython; IPython.embed()

    K = 3
    print('Testing k-anonymous nearest neighbour clustering')
    print('------------------------------------------------')

    knn = PPRLIndexKAnonymousNearestNeighbourClustering(K, dice_sim.sim,
                                                  MIN_SIM_VAL,
                                                  use_medoids=True)

    knn.load_database_alice(oz_small_alice_file_name, header_line=True,
                      rec_id_col=0, ent_id_col=0)
    knn.load_database_bob(oz_small_bob_file_name, header_line=True,
                    rec_id_col=0, ent_id_col=0)
    ref_vals = num_recs/100
    knn.load_and_select_ref_values(oz_file_name, True, oz_attr_sel_list,
                             ref_vals, random_seed=0)
    start_time = time.time()
    a_min_blk,a_med_blk,a_max_blk,a_avg_blk,a_std_dev = knn.build_index_alice(oz_attr_sel_list)
    b_min_blk,b_med_blk,b_max_blk,b_avg_blk,b_std_dev = knn.build_index_bob(oz_attr_sel_list)
    dbo_time = time.time() - start_time

    start_time = time.time()
    num_blocks = knn.generate_blocks()
    lu_time = time.time() - start_time
    rr, pc, pq, num_cand_rec_pairs = knn.assess_blocks()

    knnarisk, knnbrisk = knn.disclosure_risk()
    # draw_risk(arisk, brisk, 'KNN', alice_num_recs)


    print('Testing k-anonymous sorted neighbourhood SIZE')
    print('-----------------------------------------')


    sn = PPRLIndexKAnonymousSortedNeighbour(K, dice_sim.sim, MIN_SIM_VAL,
                                      OVERLAP, 'SIZE')

    sn.load_database_alice(oz_small_alice_file_name, header_line=True,
                     rec_id_col=0, ent_id_col=0)
    sn.load_database_bob(oz_small_bob_file_name, header_line=True,
                   rec_id_col=0, ent_id_col=0)
    sn.load_and_select_ref_values(oz_file_name, True, oz_attr_sel_list,
                            num_ref_val, random_seed=0)
    start_time = time.time()
    a_min_blk,a_med_blk,a_max_blk,a_avg_blk,a_std_dev = sn.build_index_alice(oz_attr_sel_list)
    b_min_blk,b_med_blk,b_max_blk,b_avg_blk,b_std_dev = sn.build_index_bob(oz_attr_sel_list)
    dbo_time = time.time() - start_time

    start_time = time.time()
    num_blocks = sn.generate_blocks()
    lu_time = time.time() - start_time
    rr, pc, pq, num_cand_rec_pairs = sn.assess_blocks()

    assess_results.append(['kasn',
      alice_num_recs, bob_num_recs, num_ref_val, K,
      dbo_time, lu_time, rr, pc, pq,
      a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
      b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
      num_blocks, num_cand_rec_pairs
    ])

    kasnarisk, kasnbrisk = sn.disclosure_risk()
    # draw_risk(arisk, brisk, 'KNN', alice_num_recs)

    draw_riskcompare([psigarisk, knnarisk, kasnarisk],
                     ['P-Sig', 'KNN', 'KASN'], 'Alice', alice_num_recs)
    draw_riskcompare([psigbrisk, knnbrisk, kasnbrisk],
                     ['P-Sig', 'KNN', 'KASN'], 'Bob', alice_num_recs)
