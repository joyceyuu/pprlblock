# comp_pb.py - Python routines for private indexing/blocking for PPRL
#
# Assuming we have access to data sets:
# - Australian phone book
# - North Carolina voter database
#
# Six private blocking/indexing methods:
# 1. Three-party k-nearest neighbour based clustering (Kar12 kNN)
# 2. Three-party Sorted neighbourhood clustering - SIM based merging (Vat13PAKDD - SNC3PSim)
# 3. Three-party Sorted neighbourhood clustering - SIZE based merging (Vat13PAKDD - SNC3PSize)
# 4. Three-party Bloom filter Locality Sensitive hashing based blocking (Dur12 - HLSH)
# 5. Two-party Sorted neighbourhood clustering (Vat13CIKM - SNC2P)
# 6. Two-party hclustering based blocking (Kuz13 - HCLUST)
#
#
# DV and PC
# 22/01/2014
# -----------------------

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

TEST_KANN       =  True   # k-NN clustering (Kar12 - kNN)
TEST_PSIG       =  True   # Probability Signature
TEST_KASN_SIM   =  True   # Sorted neighbourhood SIM (Vat13PAKDD - SNC3PSim)
TEST_KASN_SIZE  =  True   # Sorted neighbourhood SIZE (Vat13PAKDD - SNC3PSize)
TEST_BFLSH      =  True   # Bloom filter Locality Sensitive hashing (Dur12 - HLSH)
TEST_KASN_2P_SIM = True    # Sorted neighbourhood 2Party SIM (Vat13CIKM - SNC2P)
TEST_hClust_2P   = False   # hclustering based blocking (Kuz13 - HCLUST)

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

mod_test_mode = sys.argv[1]  # 'no', 'mod', 'lno', 'lmod', 'nc', 'syn', 'syn_mod', 'nc_syn', 'nc_syn_mod'

#####################################################################

# The pairs of data sets to be used for testing (the first one will be loaded
# and processed by Alice, the second by Bob)
#
if (mod_test_mode == 'ex'):
  data_sets_pairs = [['./datasets/example/10_25_overlap_no_mod_alice.csv',
     './datasets/example/10_25_overlap_no_mod_bob.csv']]

elif (mod_test_mode == 'no'):  # OZ No-mod
  data_sets_pairs = [
    #['./datasets/173_25_overlap_no_mod_alice.csv.gz',
    # './datasets/173_25_overlap_no_mod_bob.csv.gz'],
    #['./datasets/173_50_overlap_no_mod_alice.csv.gz',
    # './datasets/173_50_overlap_no_mod_bob.csv.gz'],
    #['./datasets/173_75_overlap_no_mod_alice.csv.gz',
    # './datasets/173_75_overlap_no_mod_bob.csv.gz'],

    #['./datasets/1730_25_overlap_no_mod_alice.csv.gz',
    # './datasets/1730_25_overlap_no_mod_bob.csv.gz'],
    # ['./datasets/1730_50_overlap_no_mod_alice.csv.gz',
    #  './datasets/1730_50_overlap_no_mod_bob.csv.gz'],
    #
    # ['./datasets/4611_50_overlap_no_mod_alice.csv',
    #  './datasets/4611_50_overlap_no_mod_bob.csv'],

    # ['./datasets/46116_50_overlap_no_mod_alice.csv',
    #  './datasets/46116_50_overlap_no_mod_bob.csv'],

    ['./datasets/461167_50_overlap_no_mod_alice.csv',
     './datasets/461167_50_overlap_no_mod_bob.csv'],
    #
    # ['./datasets/4611676_50_overlap_no_mod_alice.csv',
    #  './datasets/4611676_50_overlap_no_mod_bob.csv'],

    ]
    #['./datasets/1730_75_overlap_no_mod_alice.csv.gz',
    # './datasets/1730_75_overlap_no_mod_bob.csv.gz'],

    #['./datasets/17294_25_overlap_no_mod_alice.csv.gz',
    # './datasets/17294_25_overlap_no_mod_bob.csv.gz'],
    # ['./datasets/17294_50_overlap_no_mod_alice.csv.gz',
    #  './datasets/17294_50_overlap_no_mod_bob.csv.gz']] #,
    #['./datasets/17294_75_overlap_no_mod_alice.csv.gz',
    # './datasets/17294_75_overlap_no_mod_bob.csv.gz'],

    #['./datasets/172938_25_overlap_no_mod_alice.csv.gz',
    # './datasets/172938_25_overlap_no_mod_bob.csv.gz'],
    #['./datasets/172938_50_overlap_no_mod_alice.csv.gz',
    # './datasets/172938_50_overlap_no_mod_bob.csv.gz'],
    #['./datasets/172938_75_overlap_no_mod_alice.csv.gz',
    # './datasets/172938_75_overlap_no_mod_bob.csv.gz']]

elif (mod_test_mode == 'mod'):  # Data sets with modifications (for Bob) OZ Mod
  data_sets_pairs = [
    #['./datasets/173_25_overlap_no_mod_alice.csv.gz',
    # './datasets/173_25_overlap_with_mod_bob_1.csv.gz'],
    #['./datasets/173_50_overlap_no_mod_alice.csv.gz',
    # './datasets/173_50_overlap_with_mod_bob_1.csv.gz'],
    #['./datasets/173_75_overlap_no_mod_alice.csv.gz',
    # './datasets/173_75_overlap_with_mod_bob_1.csv.gz'],

    #['./datasets/1730_25_overlap_no_mod_alice.csv.gz',
    # './datasets/1730_25_overlap_with_mod_bob_1.csv.gz'],
    ['./datasets/1730_50_overlap_no_mod_alice.csv.gz',
     './datasets/1730_50_overlap_with_mod_bob_1.csv.gz']]
    #['./datasets/1730_75_overlap_no_mod_alice.csv.gz',
    # './datasets/1730_75_overlap_with_mod_bob_1.csv.gz'],

    #['./datasets/17294_25_overlap_no_mod_alice.csv.gz',
    # './datasets/17294_25_overlap_with_mod_bob_1.csv.gz'],
    # ['./datasets/17294_50_overlap_no_mod_alice.csv.gz',
    #  './datasets/17294_50_overlap_with_mod_bob_1.csv.gz']]#,
    #['./datasets/17294_75_overlap_no_mod_alice.csv.gz',
    # './datasets/17294_75_overlap_with_mod_bob_1.csv.gz'],

    #['./datasets/172938_25_overlap_no_mod_alice.csv.gz',
    # './datasets/172938_25_overlap_with_mod_bob_1.csv.gz'],
    #['./datasets/172938_50_overlap_no_mod_alice.csv.gz',
    # './datasets/172938_50_overlap_with_mod_bob_1.csv.gz'],
    #['./datasets/172938_75_overlap_no_mod_alice.csv.gz',
    # './datasets/172938_75_overlap_with_mod_bob_1.csv.gz']]


elif (mod_test_mode == 'lno'): # OZ largest dataset No-mod
  data_sets_pairs = [
    #['./datasets/1729379_25_overlap_no_mod_alice.csv.gz',
    # './datasets/1729379_25_overlap_no_mod_bob.csv.gz'],
    ['./datasets/1729379_50_overlap_no_mod_alice.csv.gz',
     './datasets/1729379_50_overlap_no_mod_bob.csv.gz']]
    #['./datasets/1729379_75_overlap_no_mod_alice.csv.gz',
    # './datasets/1729379_75_overlap_no_mod_bob.csv.gz']]

elif (mod_test_mode == 'lmod'): # OZ largest dataset mod
  data_sets_pairs = [
    #['./datasets/1729379_25_overlap_no_mod_alice.csv.gz',
    # './datasets/1729379_25_overlap_with_mod_bob_1.csv.gz'],
    ['./datasets/1729379_50_overlap_no_mod_alice.csv.gz',
     './datasets/1729379_50_overlap_with_mod_bob_1.csv.gz']]
    #['./datasets/1729379_75_overlap_no_mod_alice.csv.gz',
    # './datasets/1729379_75_overlap_with_mod_bob_1.csv.gz']]

elif (mod_test_mode == 'nc'): # NC dataset
  data_sets_pairs = [
    ['./datasets/ncvoter-temporal-1.csv',
     './datasets/ncvoter-temporal-2.csv']]

elif (mod_test_mode == 'syn'): # OZ Cor No-mod
  data_sets_pairs = [
    ['./datasets/4611_50_overlap_no_mod_alice.csv',
     './datasets/4611_50_overlap_no_mod_bob.csv'],
    ['./datasets/46116_50_overlap_no_mod_alice.csv',
     './datasets/46116_50_overlap_no_mod_bob.csv'],
    ['./datasets/461167_50_overlap_no_mod_alice.csv',
     './datasets/461167_50_overlap_no_mod_bob.csv']]#,
    #['./datasets/4611676_50_overlap_no_mod_alice.csv',
    # './datasets/4611676_50_overlap_no_mod_bob.csv']]

elif (mod_test_mode == 'syn_mod'): # OZ Cor Light-mod, Med-mod, and Heavy-mod
  data_sets_pairs = [
    ['./datasets/4611_50_overlap_no_mod_alice.csv',
     './datasets/4611_50_overlap_with_mod_bob_1.csv'],
    ['./datasets/4611_50_overlap_no_mod_alice.csv',
     './datasets/4611_50_overlap_with_mod_bob_2.csv'],
    ['./datasets/4611_50_overlap_no_mod_alice.csv',
     './datasets/4611_50_overlap_with_mod_bob_4.csv'],
    ['./datasets/46116_50_overlap_no_mod_alice.csv',
     './datasets/46116_50_overlap_with_mod_bob_1.csv'],
    ['./datasets/46116_50_overlap_no_mod_alice.csv',
     './datasets/46116_50_overlap_with_mod_bob_2.csv'],
    ['./datasets/46116_50_overlap_no_mod_alice.csv',
     './datasets/46116_50_overlap_with_mod_bob_4.csv'],
    ['./datasets/461167_50_overlap_no_mod_alice.csv',
     './datasets/461167_50_overlap_with_mod_bob_1.csv'],
    ['./datasets/461167_50_overlap_no_mod_alice.csv',
     './datasets/461167_50_overlap_with_mod_bob_2.csv'],
    ['./datasets/461167_50_overlap_no_mod_alice.csv',
     './datasets/461167_50_overlap_with_mod_bob_4.csv']] #,
    #['./datasets/4611676_50_overlap_no_mod_alice.csv',
    # './datasets/4611676_50_overlap_with_mod_bob_1.csv'],
    #['./datasets/4611676_50_overlap_no_mod_alice.csv',
    # './datasets/4611676_50_overlap_with_mod_bob_2.csv'],
    #['./datasets/4611676_50_overlap_no_mod_alice.csv',
    # './datasets/4611676_50_overlap_with_mod_bob_4.csv']]

elif (mod_test_mode == 'nc_syn'): # NC Cor No-mod
  data_sets_pairs = [
    ['./datasets/5488_50_overlap_no_mod_alice.csv',
     './datasets/5488_50_overlap_no_mod_bob.csv'],
    ['./datasets/54886_50_overlap_no_mod_alice.csv',
     './datasets/54886_50_overlap_no_mod_bob.csv'],
    ['./datasets/548860_50_overlap_no_mod_alice.csv',
     './datasets/548860_50_overlap_no_mod_bob.csv']]#,

elif (mod_test_mode == 'nc_syn_mod'): # NC Cor Light-mod, Med-mod, and Heavy-mod
  data_sets_pairs = [
    ['./datasets/5488_50_overlap_no_mod_alice.csv',
     './datasets/5488_50_overlap_with_mod_bob_1.csv'],
    ['./datasets/5488_50_overlap_no_mod_alice.csv',
     './datasets/5488_50_overlap_with_mod_bob_2.csv'],
    ['./datasets/5488_50_overlap_no_mod_alice.csv',
     './datasets/5488_50_overlap_with_mod_bob_4.csv'],
    ['./datasets/54886_50_overlap_no_mod_alice.csv',
     './datasets/54886_50_overlap_with_mod_bob_1.csv'],
    ['./datasets/54886_50_overlap_no_mod_alice.csv',
     './datasets/54886_50_overlap_with_mod_bob_2.csv'],
    ['./datasets/54886_50_overlap_no_mod_alice.csv',
     './datasets/54886_50_overlap_with_mod_bob_4.csv'],
    ['./datasets/548860_50_overlap_no_mod_alice.csv',
     './datasets/548860_50_overlap_with_mod_bob_1.csv'],
    ['./datasets/548860_50_overlap_no_mod_alice.csv',
     './datasets/548860_50_overlap_with_mod_bob_2.csv'],
    ['./datasets/548860_50_overlap_no_mod_alice.csv',
     './datasets/548860_50_overlap_with_mod_bob_4.csv']] #,


####################################################################

#oz_small_alice_file_name = 'datasets/17294_25_overlap_no_mod_alice.csv.gz'
#oz_small_bob_file_name =   'datasets/17294_25_overlap_no_mod_bob.csv.gz'

#
# ============================================================================
# Main program

# Attributes:
# Oz: 1=gname,2=surname,3=suburb,4=postcode
# NC: 3=fname,5=lname,12=city,14=zipcode
#

oz_attr_sel_list = [1,2]
#nc_attr_sel_list = [3,5]

attr_bf_sample_list = [60,40]  # Sample 50% of bits from attribute BF
dice_sim = DiceSim()

bf_sim =   BloomFilterSim()

# ----------------------------------------------------------------------------

def write_results(out_file_name, alice_num_recs, bob_num_recs, num_ref_val, K,
                  dbo_time, lu_time, rr, pc, pq,
                  a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
                  b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
                  num_blocks, num_cand_rec_pairs,
                  algorithem_name
                  ):
    """Write the results to file for blocking method."""
    if os.path.exists(out_file_name):
        out_file = open(out_file_name, 'a')
    else:
        out_file = open(out_file_name, 'w')
        out_file.write('alice_file,bob_file,num_ref_val,k,time,'+\
                        'rr,pc,pq,alice_min_blk,alice_med_blk,'+\
                        'alice_max_blk,alice_avg_blk,bob_min_blk,'+\
                        'bob_med_blk,bob_max_blk,bob_avg_blk,'+\
                        'num_blocks,num_cand_pairs,num_blocks,num_cand_rec_pairs'+os.linesep)
    log_str = str(alice_num_recs)+','+str(bob_num_recs)+','\
               +str(num_ref_val)+','+str(K)
    log_str += ',%.4f,%.4f,%.4f,%.4f,%.4f' % \
    (dbo_time,lu_time, rr,pc,pq)
    log_str += ',%i,%.2f,%i,%.2f,%.2f' % \
    (a_min_blk,a_med_blk,a_max_blk,a_avg_blk,a_std_dev)
    log_str += ',%i,%.2f,%i,%.2f,%.2f' % \
    (b_min_blk,b_med_blk,b_max_blk,b_avg_blk,b_std_dev)
    log_str += ','+str(num_blocks)
    log_str += ','+str(num_cand_rec_pairs)
    out_file.write(log_str+os.linesep)
    out_file.close()

    # print('Quality and complexity results for sorted neighbourhood SIM:')

    print('Quality and complexity results for {}:'.format(algorithem_name))
    print('  rr = %.4f' % (rr))
    print('  pc = %.4f' % (pc))
    print('  pq = %.4f' % (pq))
    print('  dbo time = %.4f' % (dbo_time))
    print('  lu time = %.4f' % (lu_time))



for K in [100]: #[3,10,20,50,100]:

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

    if (TEST_KANN == True):

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


        assess_results.append(['knn',
            alice_num_recs, bob_num_recs, num_ref_val, K,
            dbo_time, lu_time, rr, pc, pq,
            a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
            b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
            num_blocks, num_cand_rec_pairs
        ])
        write_results('./logs/kNN.csv', alice_num_recs, bob_num_recs, num_ref_val, K,
                      dbo_time, lu_time, rr, pc, pq,
                      a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
                      b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
                      num_blocks, num_cand_rec_pairs,
                      'k-nearest neighbour clustering'
                      )

# ----------------------------------------------------------------------------
    if TEST_PSIG == True:

        print('Testing probability signature')
        print('------------------------------------------------')

        psig = PPRLIndexPSignature(num_hash_funct=20, bf_len=1024)
        psig.load_database_alice(oz_small_alice_file_name, header_line=True,
                               rec_id_col=0, ent_id_col=0)
        psig.load_database_bob(oz_small_bob_file_name, header_line=True,
                               rec_id_col=0, ent_id_col=0)
        start_time = time.time()
        psig.common_bloom_filter([1, 2])
        psig.drop_toofrequent_index(len(psig.rec_dict_alice) * 0.04)
        a_min_blk,a_med_blk,a_max_blk,a_avg_blk,a_std_dev = psig.build_index_alice()
        b_min_blk,b_med_blk,b_max_blk,b_avg_blk,b_std_dev = psig.build_index_bob()
        dbo_time = time.time() - start_time

        start_time = time.time()
        num_blocks = psig.generate_blocks()
        lu_time = time.time() - start_time
        rr, pc, pq, num_cand_rec_pairs = psig.assess_blocks()
        assess_results.append(['psig',
            alice_num_recs, bob_num_recs, num_ref_val, K,
            dbo_time, lu_time, rr, pc, pq,
            a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
            b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
            num_blocks, num_cand_rec_pairs
        ])

        write_results('./logs/PSig.csv', alice_num_recs, bob_num_recs, num_ref_val, K,
                      dbo_time, lu_time, rr, pc, pq,
                      a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
                      b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
                      num_blocks, num_cand_rec_pairs, 'probability signature'
                      )

# ----------------------------------------------------------------------------

    if (TEST_KASN_SIM == True):
      print()
      print()

      print('Testing k-anonymous sorted neighbourhood SIM')
      print('-----------------------------------------')

      sn = PPRLIndexKAnonymousSortedNeighbour(K, dice_sim.sim, MIN_SIM_VAL,
                                          OVERLAP, 'SIM')

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

      assess_results.append(['snn_sim',
          alice_num_recs, bob_num_recs, num_ref_val, K,
          dbo_time, lu_time, rr, pc, pq,
          a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
          b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
          num_blocks, num_cand_rec_pairs
      ])

      write_results('./logs/SNN_SIM.csv', alice_num_recs, bob_num_recs, num_ref_val, K,
                      dbo_time, lu_time, rr, pc, pq,
                      a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
                      b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
                      num_blocks, num_cand_rec_pairs, 'sorted neighbourhood SIM'
                      )



# ----------------------------------------------------------------------------

    if (TEST_KASN_SIZE == True):
      print()
      print()

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

      write_results('./logs/KASN.csv', alice_num_recs, bob_num_recs, num_ref_val, K,
                      dbo_time, lu_time, rr, pc, pq,
                      a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
                      b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
                      num_blocks, num_cand_rec_pairs, 'sorted neighbourhood SIZE'
                      )


# ----------------------------------------------------------------------------

    if (TEST_BFLSH == True):
      K = 100
      print()
      print()

      print('Testing Bloom filter Hamming LSH')
      print('--------------------------------')

      bf = PPRLIndexBloomFilterHLSH(N_HASH, SET_BIT_PERC, RAND_SEED)

      bf.load_database_alice(oz_small_alice_file_name, header_line=True,
                         rec_id_col=0, ent_id_col=0)
      bf.load_database_bob(oz_small_bob_file_name, header_line=True,
                       rec_id_col=0, ent_id_col=0)
      start_time = time.time()
      a_min_blk,a_med_blk,a_max_blk,a_avg_blk,a_std_dev = \
         bf.build_index_alice(oz_attr_sel_list,attr_bf_sample_list, HLSH_NUM_BIT, HLSH_NUM_ITER)
      b_min_blk,b_med_blk,b_max_blk,b_avg_blk,b_std_dev = \
         bf.build_index_bob(oz_attr_sel_list,attr_bf_sample_list, HLSH_NUM_BIT, HLSH_NUM_ITER)
      dbo_time = time.time() - start_time

      start_time = time.time()
      num_blocks = bf.generate_blocks()
      lu_time = time.time() - start_time
      rr, pc, pq, num_cand_rec_pairs = bf.assess_blocks()

      assess_results.append(['hlsh_clust',
          alice_num_recs, bob_num_recs, num_ref_val, K,
          dbo_time, lu_time, rr, pc, pq,
          a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
          b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
          num_blocks, num_cand_rec_pairs
      ])

      write_results('./logs/HLSH_clust.csv', alice_num_recs, bob_num_recs, num_ref_val, K,
                      dbo_time, lu_time, rr, pc, pq,
                      a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
                      b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
                      num_blocks, num_cand_rec_pairs, 'Bloom filter Hamming LSH'
                      )


# ----------------------------------------------------------------------------

    if (TEST_KASN_2P_SIM == True):
      K = 100
      print()
      print()

      print('Testing k-anonymous 2-party sorted neighbourhood SIM')
      print('-----------------------------------------')

      sn = PPRLIndex2PartyKAnonymousSortedNeighbour(K, W, dice_sim.sim, MIN_SIM_VAL,
                                          OVERLAP, 'SIM')

      sn.load_database_alice(oz_small_alice_file_name, header_line=True,
                         rec_id_col=0, ent_id_col=0)
      sn.load_database_bob(oz_small_bob_file_name, header_line=True,
                       rec_id_col=0, ent_id_col=0)
      R = 10 #40
      ref_vals = num_recs/K * R
      sn.load_and_select_ref_values_alice(oz_file_name, True, oz_attr_sel_list,
                                ref_vals, random_seed=1)
      sn.load_and_select_ref_values_bob(oz_file_name, True, oz_attr_sel_list,
                                ref_vals, random_seed=0)
      start_time = time.time()
      a_min_blk,a_med_blk,a_max_blk,a_avg_blk,a_std_dev,alice_time = sn.build_index_alice(oz_attr_sel_list)
      b_min_blk,b_med_blk,b_max_blk,b_avg_blk,b_std_dev,bob_time = sn.build_index_bob(oz_attr_sel_list)
      dbo_time = time.time() - start_time


      start_time = time.time()
      #sn.generate_blocks()
      num_blocks,block_time = sn.generate_blocks()
      lu_time = time.time() - start_time
      rr, pc, pq, num_cand_rec_pairs = sn.assess_blocks()

      assess_results.append(['snc2p',
          alice_num_recs, bob_num_recs, num_ref_val, K,
          dbo_time, lu_time, rr, pc, pq,
          a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
          b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
          num_blocks, num_cand_rec_pairs
      ])

      tot_time = max(alice_time,bob_time)+block_time
      #tot_time = dbo_time + lu_time

      write_results('./logs/SNC2P.csv', alice_num_recs, bob_num_recs, num_ref_val, K,
                      dbo_time, lu_time, rr, pc, pq,
                      a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
                      b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
                      num_blocks, num_cand_rec_pairs, '2-party sorted neighbourhood SIM'
                      )

# ----------------------------------------------------------------------------

    if (TEST_hClust_2P == True):

      K = 100
      print()
      print()

      print('Testing 2-party h-clustering')
      print('-----------------------------------------')

      # out_file_name = './logs/hclust.csv'
      # out_file = open(out_file_name, 'a')
      # out_file.write('hclust'+os.linesep)
      # out_file.write('alice_file,bob_file,num_ref_val,time,'+\
      #               'rr,pc,pq,alice_min_blk,alice_med_blk,'+\
      #               'alice_max_blk,alice_avg_blk,bob_min_blk,'+\
      #               'bob_med_blk,bob_max_blk,bob_avg_blk,'+\
      #               'num_blocks,num_cand_pairs'+os.linesep)
      #
      # log_str = str(alice_num_recs)+','+str(bob_num_recs)+','\
      #              +str(num_ref_val)

      hc = hclustering(editdist, nb=num_recs/10, wn=num_recs, ep=0.3)

      hc.load_database_alice(oz_small_alice_file_name, header_line=True,
                         rec_id_col=0, ent_id_col=0)
      hc.load_database_bob(oz_small_bob_file_name, header_line=True,
                       rec_id_col=0, ent_id_col=0)
      ref_vals = num_recs/100
      hc.load_and_select_ref_values(oz_file_name, True, oz_attr_sel_list,
                                ref_vals, random_seed=1)

      start_time = time.time()
      clust = hc.hcluster()
      a_min_blk,a_med_blk,a_max_blk,a_avg_blk,a_std_dev,alice_time,au_list = hc.build_index_alice(oz_attr_sel_list,clust)
      b_min_blk,b_med_blk,b_max_blk,b_avg_blk,b_std_dev,bob_time,bu_list = hc.build_index_bob(oz_attr_sel_list,clust)
      dbo_time = time.time() - start_time


      start_time = time.time()
      #sn.generate_blocks()
      num_blocks = hc.generate_blocks()
      lu_time = time.time() - start_time
      rr, pc, pq, num_cand_rec_pairs = hc.assess_blocks()

      assess_results.append(['hclust',
          alice_num_recs, bob_num_recs, num_ref_val, K,
          dbo_time, lu_time, rr, pc, pq,
          a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
          b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
          num_blocks, num_cand_rec_pairs
      ])

      write_results('./logs/hclust.csv', alice_num_recs, bob_num_recs, num_ref_val, K,
                        dbo_time, lu_time, rr, pc, pq,
                        a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
                        b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
                        num_blocks, num_cand_rec_pairs, '2-party hclustering'
                        )
      #tot_time = max(alice_time,bob_time)+block_time
      tot_time = dbo_time + lu_time

      out_file_name = './logs/hclust_noise.csv'
      out_file = open(out_file_name, 'a')
      out_file.write(str(num_recs))
      out_file.write(str(au_list))
      out_file.write(os.linesep)
      out_file.write(str(bu_list))
      out_file.write(os.linesep)
      print(au_list, bu_list)
      out_file.close()

    # dataframe that summarize all methods
    df = pd.DataFrame(data=assess_results)
    df.columns = ['Method',
        'alice_num_recs', 'bob_num_recs', 'num_ref_val', 'K',
        'dbo_time', 'lu_time', 'rr', 'pc', 'pq',
        'a_min_blk', 'a_med_blk', 'a_max_blk', 'a_avg_blk', 'a_std_dev',
        'b_min_blk', 'b_med_blk', 'b_max_blk', 'b_avg_blk', 'b_std_dev',
        'num_blocks', 'num_cand_rec_pairs']
    print()
    print(df)
    n = df['alice_num_recs'].unique()[0]
    df.to_csv('result_n={}.csv'.format(n), index=False)
import IPython; IPython.embed()
