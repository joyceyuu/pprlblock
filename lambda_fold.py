from blocklib import PPRLIndexLambdaFold
from blocklib import generate_blocks, assess_blocks_2party, generate_candidate_blocks
import time
import pandas as pd
import numpy as np
import pandas as pd


data_sets_pairs = [
    # ['./datasets/4611_50_overlap_no_mod_alice.csv',
    #  './datasets/4611_50_overlap_no_mod_bob.csv'],

    # ['./datasets/46116_50_overlap_no_mod_alice.csv',
    #  './datasets/46116_50_overlap_no_mod_bob.csv'],

    ['./datasets/461167_50_overlap_no_mod_alice.csv',
     './datasets/461167_50_overlap_no_mod_bob.csv'],

    # ['./datasets/4611676_50_overlap_no_mod_alice.csv',
    #  './datasets/4611676_50_overlap_no_mod_bob.csv'],

]


config = {
    "type": "lambda-fold",
    "version": 1,
    "config": {
        "blocking-features": [1, 2],
        "Lambda": 5,
        "bf-len": 2000,
        "num-hash-funcs": 10,
        "K": 40,
        "random_state": 0
    }
    # "type": "p-sig",
    # "version": 1,
    # "config": {
    #     "blocking_features": [1],
    #     # "record-id-col": 0,
    #     "filter": {
    #         "type": "ratio",
    #         "max": 0.02,
    #         "min": 0.00,
    #     },
    #     "blocking-filter": {
    #         "type": "bloom filter",
    #         "number-hash-functions": 4,
    #         "bf-len": 2048,
    #     },
    #     "signatureSpecs": [
    #         [
    #              {"type": "characters-at", "config": {"pos": [0]}, "feature-idx": 1},
    #              {"type": "characters-at", "config": {"pos": [0]}, "feature-idx": 2},
    #         ],
    #         [
    #             {"type": "characters-at", "config": {"pos": [0]}, "feature-idx": 1},
    #             {"type": "characters-at", "config": {"pos": [1]}, "feature-idx": 1},
    #         ],
    #         [
    #             {"type": "characters-at", "config": {"pos": [0]}, "feature-idx": 2},
    #             {"type": "characters-at", "config": {"pos": [1]}, "feature-idx": 2},
    #         ],
    #         [
    #             {"type": "characters-at", "config": {"pos": [":2"]}, "feature-idx": 3},
    #         ],
    #         [
    #             {"type": "metaphone", "feature-idx": 1},
    #             {"type": "metaphone", "feature-idx": 2},
    #         ]
    #     ]
    # }
}

# load data
file_alice, file_bob = data_sets_pairs[0]
num_recs_alice = int(file_alice.split('/')[-1].split('_')[0])
num_recs_bob = int(file_bob.split('/')[-1].split('_')[0])


print('Loading dataset Alice n={}'.format(num_recs_alice))
alice_data = pd.read_csv(file_alice)
alice_data = alice_data.to_dict(orient='split')['data']

print('Loading dataset Bob n={}'.format(num_recs_bob))
bob_data = pd.read_csv(file_alice)
bob_data = bob_data.to_dict(orient='split')['data']
print('Example data = {}'.format(alice_data[0]))

# build candidate blocks
start_time = time.time()
print('Building reversed index of Alice')
block_obj_alice = generate_candidate_blocks(alice_data, config)

print('Building reversed index of Bob')
block_obj_bob = generate_candidate_blocks(bob_data, config)
dbo_time = time.time() - start_time

# gather statistics
a_min_blk = block_obj_alice.state.stats['min_size']
a_med_blk = block_obj_alice.state.stats['med_size']
a_max_blk = block_obj_alice.state.stats['max_size']
a_avg_blk = block_obj_alice.state.stats['avg_size']
a_std_dev = block_obj_alice.state.stats['std_size']

b_min_blk = block_obj_bob.state.stats['min_size']
b_med_blk = block_obj_bob.state.stats['med_size']
b_max_blk = block_obj_bob.state.stats['max_size']
b_avg_blk = block_obj_bob.state.stats['avg_size']
b_std_dev = block_obj_bob.state.stats['std_size']

# build final blocks
start_time = time.time()
print('Filtering reversed index - Generate final blocks')
filtered_alice, filtered_bob = generate_blocks([block_obj_alice, block_obj_bob], K=2)
lu_time = time.time() - start_time
num_blocks = len(filtered_alice)

# assess
subdata1 = [[x[0]] for x in alice_data]
subdata2 = [[x[0]] for x in bob_data]

rr, pc = assess_blocks_2party([filtered_alice, filtered_bob], [subdata1, subdata2])

method_name = config['type']
result = dict(Method=method_name, alice_num_recs=num_recs_alice, bob_num_recs=num_recs_bob, num_ref_val=None,
              K=3, rr=rr, pc=pc, pq=None, dbo_time=dbo_time, lu_time=lu_time, assess_time=None,
              tot_time=dbo_time + lu_time, a_min_blk=a_min_blk, a_med_blk=a_med_blk, a_max_blk=a_max_blk,
              a_avg_blk=a_avg_blk, a_std_dev=a_std_dev, b_min_blk=b_min_blk, b_med_blk=b_med_blk, b_max_blk=b_max_blk,
              b_avg_blk=b_avg_blk, b_std_dev=b_std_dev, num_blocks=num_blocks, num_cand_rec_pairs=100)

result = {k: [v] for k, v in result.items()}
df = pd.DataFrame.from_dict(result)
df.columns = ['Method',
              'alice_num_recs', 'bob_num_recs', 'num_ref_val', 'K',
              'rr', 'pc', 'pq', 'dbo_time', 'lu_time', 'assess_time', 'tot_time',
              'a_min_blk', 'a_med_blk', 'a_max_blk', 'a_avg_blk', 'a_std_dev',
              'b_min_blk', 'b_med_blk', 'b_max_blk', 'b_avg_blk', 'b_std_dev',
              'num_blocks', 'num_cand_rec_pairs']

ddf = pd.read_csv('result2_n={}.csv'.format(num_recs_alice))
ddf = ddf[ddf['Method'] != method_name]
print(ddf)
finaldf = pd.concat([ddf, df], axis=0)
print(finaldf)
finaldf.to_csv('result2_n={}.csv'.format(num_recs_alice), index=False)