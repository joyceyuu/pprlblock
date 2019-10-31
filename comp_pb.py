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

import sys
import os
import time
import pandas as pd
from numpy import *

from pprl2partyknnsorted import PPRLIndex2PartyKAnonymousSortedNeighbour
from pprlbloomfilterhlsh import PPRLIndexBloomFilterHLSH
from pprlhclustering import hclustering
from pprlknn import PPRLIndexKAnonymousNearestNeighbourClustering
from pprlknnsorted import PPRLIndexKAnonymousSortedNeighbour
from pprlpsig import PPRLIndexPSignature
from simmeasure import DiceSim, BloomFilterSim, editdist
from get_experiment_data import experiment_data

BLOCKING_METHODS = [
    'KNN',  # k-NN clustering (Kar12 - kNN)
    # 'PSIG',  # Probability Signature
    'KASN_SIM',  # Sorted neighbourhood SIM (Vat13PAKDD - SNC3PSim)
    'KASN_SIZE',  # Sorted neighbourhood SIZE (Vat13PAKDD - SNC3PSize)
    'BFLSH',  # Bloom filter Locality Sensitive hashing (Dur12 - HLSH)
    'KASN_2P_SIM',  # Sorted neighbourhood 2Party SIM (Vat13CIKM - SNC2P)
    'HCLUST_2P'  # hclustering based blocking (Kuz13 - HCLUST)
]

MIN_SIM_VAL = 0.8  # 0.9 , 0.8, 0.6 - SNN3P, SNN2P, kNN
W = 2  # 4,7,8
OVERLAP = 0

N_HASH = 30
SET_BIT_PERC = 50
RAND_SEED = 42

HLSH_NUM_BIT = 45
HLSH_NUM_ITER = 40

oz_file_name = 'datasets/OZ-clean-with-gname.csv'

mod_test_mode = sys.argv[1]  # 'no', 'mod', 'lno', 'lmod', 'nc', 'syn', 'syn_mod', 'nc_syn', 'nc_syn_mod'
data_sets_pairs = experiment_data(mod_test_mode)

# ============================================================================
# Main program

# Attributes:
# Oz: 1=gname,2=surname,3=suburb,4=postcode
# NC: 3=fname,5=lname,12=city,14=zipcode
#

OZ_ATTR_SEL_LIST = [1, 2]

ATTR_BF_SAMPLE_LIST = [60, 40]  # Sample 50% of bits from attribute BF
dice_sim = DiceSim()

bf_sim = BloomFilterSim()


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
        out_file.write('alice_file,bob_file,num_ref_val,k,time,' + \
                       'rr,pc,pq,alice_min_blk,alice_med_blk,' + \
                       'alice_max_blk,alice_avg_blk,bob_min_blk,' + \
                       'bob_med_blk,bob_max_blk,bob_avg_blk,' + \
                       'num_blocks,num_cand_pairs,num_blocks,num_cand_rec_pairs' + os.linesep)
    log_str = str(alice_num_recs) + ',' + str(bob_num_recs) + ',' \
              + str(num_ref_val) + ',' + str(K)
    log_str += ',%.4f,%.4f,%.4f,%.4f,%.4f' % \
               (dbo_time, lu_time, rr, pc, pq)
    log_str += ',%i,%.2f,%i,%.2f,%.2f' % \
               (a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev)
    log_str += ',%i,%.2f,%i,%.2f,%.2f' % \
               (b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev)
    log_str += ',' + str(num_blocks)
    log_str += ',' + str(num_cand_rec_pairs)
    out_file.write(log_str + os.linesep)
    out_file.close()

    # print('Quality and complexity results for sorted neighbourhood SIM:')

    print('Quality and complexity results for {}:'.format(algorithem_name))
    print('  rr = %.4f' % (rr))
    print('  pc = %.4f' % (pc))
    print('  pq = %.4f' % (pq))
    print('  dbo time = %.4f' % (dbo_time))
    print('  lu time = %.4f' % (lu_time))


def experiment(pprlclass, alice_data_file, bob_data_file, load_ref_data, ref_config, assess_results,
               name, name_short, args, build_index_args):
    """Experiment the PPRL blocking technique."""
    print()
    print()
    print('Testing ', name)
    print('------------------------------------------------')
    obj = pprlclass(**args)

    # load data
    obj.load_database_alice(alice_data_file, header_line=True, rec_id_col=0, ent_id_col=0)
    obj.load_database_bob(bob_data_file, header_line=True, rec_id_col=0, ent_id_col=0)

    # load reference data if needed
    if load_ref_data:
        two_party = ref_config['two_party']
        ref_vals = ref_config['ref_vals']
        num_ref_val = ref_config['num_ref_val']
        ref_data_file = ref_config['ref_data_file']
        if two_party:
            obj.load_and_select_ref_values_alice(ref_data_file, True, OZ_ATTR_SEL_LIST, ref_vals, random_seed=1)
            obj.load_and_select_ref_values_bob(ref_data_file, True, OZ_ATTR_SEL_LIST, ref_vals, random_seed=0)
        else:
            obj.load_and_select_ref_values(ref_data_file, True, OZ_ATTR_SEL_LIST, ref_vals, random_seed=0)

    else:
        num_ref_val = None

    # start to build reverse index of alice and bob
    start_time = time.time()
    a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev = obj.build_index_alice(OZ_ATTR_SEL_LIST, **build_index_args)
    b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev = obj.build_index_bob(OZ_ATTR_SEL_LIST, **build_index_args)
    dbo_time = time.time() - start_time

    # start to build blocks
    start_time = time.time()
    num_blocks = obj.generate_blocks()
    lu_time = time.time() - start_time

    # start to assess blocks
    start_time = time.time()
    rr, pc, pq, num_cand_rec_pairs = obj.assess_blocks()
    assess_time = time.time() - start_time

    tot_time = dbo_time + lu_time + assess_time

    assess_results.append([name_short,
                           alice_num_recs, bob_num_recs, num_ref_val, K,
                           rr, pc, pq,  dbo_time, lu_time, assess_time, tot_time,
                           a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
                           b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
                           num_blocks, num_cand_rec_pairs
                           ])
    write_results('./logs/{}.csv'.format(name_short), alice_num_recs, bob_num_recs, num_ref_val, K,
                  dbo_time, lu_time, rr, pc, pq,
                  a_min_blk, a_med_blk, a_max_blk, a_avg_blk, a_std_dev,
                  b_min_blk, b_med_blk, b_max_blk, b_avg_blk, b_std_dev,
                  num_blocks, num_cand_rec_pairs,
                  name
                  )


for K in [100]:  # [3,10,20,50,100]:

    for (alice_data_set, bob_data_set) in data_sets_pairs:

        assess_results = []
        oz_small_alice_file_name = alice_data_set
        oz_small_bob_file_name = bob_data_set

        # num of reference values R = N/k
        alice_dataset_str = alice_data_set.split('/')[-1]
        alice_num_recs = int(alice_dataset_str.split('_')[0])

        bob_dataset_str = bob_data_set.split('/')[-1]
        bob_num_recs = int(bob_dataset_str.split('_')[0])

        num_recs = max(alice_num_recs, bob_num_recs)
        num_ref_val = num_recs / K

        # config for blocking that needs reference data
        ref_config = {'ref_data_file': oz_file_name,
                      'num_ref_val': num_ref_val,
                      'ref_vals': num_recs / K,
                      'two_party': False}

        ATTR_BF_SAMPLE_LIST, HLSH_NUM_BIT, HLSH_NUM_ITER
        if 'KNN' in BLOCKING_METHODS:
            K = 3
            args = dict(k=K, sim_measure=dice_sim.sim, min_sim_threshold=MIN_SIM_VAL, use_medoids=True)
            experiment(PPRLIndexKAnonymousNearestNeighbourClustering, oz_small_alice_file_name, oz_small_bob_file_name,
                       True, ref_config, assess_results, 'k-anonymous nearest neighbour clustering', 'knn', args, {})

        # ----------------------------------------------------------------------------
        if 'PSIG' in BLOCKING_METHODS:
            args = dict(num_hash_funct=20, bf_len=2048, gram_n=3)
            experiment(PPRLIndexPSignature, oz_small_alice_file_name, oz_small_bob_file_name,
                       False, None, assess_results, 'probability signature', 'psig', args, {})

        # ----------------------------------------------------------------------------
        if 'KASN_SIM' in BLOCKING_METHODS:
            args = dict(k=K, sim_measure=dice_sim.sim, min_sim_threshold=MIN_SIM_VAL, overlap=OVERLAP,
                        sim_or_size='SIM')
            experiment(PPRLIndexKAnonymousSortedNeighbour, oz_small_alice_file_name, oz_small_bob_file_name,
                       True, ref_config, assess_results, 'k-anonymous sorted neighbourhood SIM', 'kasn_sim', args, {})

        # ----------------------------------------------------------------------------
        if 'KASN_SIZE' in BLOCKING_METHODS:
            print()
            print()
            args = dict(k=K, sim_measure=dice_sim.sim, min_sim_threshold=MIN_SIM_VAL, overlap=OVERLAP,
                        sim_or_size='SIZE')
            experiment(PPRLIndexKAnonymousSortedNeighbour, oz_small_alice_file_name, oz_small_bob_file_name,
                       True, ref_config, assess_results, 'k-anonymous sorted neighbourhood SIZE', 'kasn_size', args, {})

        # ----------------------------------------------------------------------------
        if 'BFLSH' in BLOCKING_METHODS:
            args = dict(num_hash_funct=N_HASH, one_bit_set_perc=SET_BIT_PERC, random_seed=RAND_SEED)
            build_index_args = dict(attr_bf_sample_list=ATTR_BF_SAMPLE_LIST,num_bits_hlsh=HLSH_NUM_BIT,
                                    num_iter_hlsh=HLSH_NUM_ITER)
            experiment(PPRLIndexBloomFilterHLSH, oz_small_alice_file_name, oz_small_bob_file_name,
                       False, None, assess_results, 'Bloom filter Hamming LSH', 'bflsh_clust', args, build_index_args)

        # ----------------------------------------------------------------------------

        if 'KASN_2P_SIM' in BLOCKING_METHODS:
            args = dict(k=K, W=W, sim_measure=dice_sim.sim, min_sim_threshold=MIN_SIM_VAL, overlap=OVERLAP,
                        sim_or_size='SIZE')
            ref_config['two_party'] = True
            experiment(PPRLIndex2PartyKAnonymousSortedNeighbour, oz_small_alice_file_name, oz_small_bob_file_name,
                       True, ref_config, assess_results, 'k-anonymous 2-party sorted neighbourhood SIM', 'snc2p', args, {})

        # ----------------------------------------------------------------------------

        if 'HCLUST_2P' in BLOCKING_METHODS:
            args = dict(dist=editdist, nb=num_recs/10, wn=num_recs, ep=0.3)
            experiment(hclustering, oz_small_alice_file_name, oz_small_bob_file_name,
                       False, None, assess_results, '2-party hclustering', 'hclust', args, {})

        # dataframe that summarize all methods
        df = pd.DataFrame(data=assess_results)
        df.columns = ['Method',
                      'alice_num_recs', 'bob_num_recs', 'num_ref_val', 'K',
                      'rr', 'pc', 'pq', 'dbo_time', 'lu_time', 'assess_time', 'tot_time',
                      'a_min_blk', 'a_med_blk', 'a_max_blk', 'a_avg_blk', 'a_std_dev',
                      'b_min_blk', 'b_med_blk', 'b_max_blk', 'b_avg_blk', 'b_std_dev',
                      'num_blocks', 'num_cand_rec_pairs']
        print()
        print(df)
        n = df['alice_num_recs'].unique()[0]
        df.to_csv('result_n={}.csv'.format(n), index=False)
