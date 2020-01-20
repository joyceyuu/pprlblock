

from blocklib import PPRLIndexLambdaFold
from blocklib import generate_blocks, assess_blocks_2party, generate_candidate_blocks
import time
import math
import pandas as pd
import numpy as np
import pandas as pd
import sys
from vis import corr_rg_rr
mod = True

data_sets_pairs = [
    ['./datasets/4611_50_overlap_no_mod_alice.csv',
     './datasets/4611_50_overlap_with_mod_bob_1.csv'],

    # ['./datasets/46116_50_overlap_no_mod_alice.csv',
    #  './datasets/46116_50_overlap_no_mod_bob.csv'],
    #
    # ['./datasets/461167_50_overlap_no_mod_alice.csv',
    #  './datasets/461167_50_overlap_no_mod_bob.csv'],

    # ['./datasets/4611676_50_overlap_no_mod_alice.csv',
    #  './datasets/4611676_50_overlap_no_mod_bob.csv'],

]


def experiment():
    max_ratio = np.linspace(0.01, 0.02, 20)
    config = {
        "type": "p-sig",
        "version": 1,
        "config": {
            "blocking_features": [1],
            # "record-id-col": 0,
            "filter": {
                "type": "ratio",
                "max": 0.02,
                "min": 0.00,
            },
            "blocking-filter": {
                "type": "bloom filter",
                "number-hash-functions": 4,
                "bf-len": 2048,
            },
            "signatureSpecs": [
                [
                     {"type": "characters-at", "config": {"pos": [0]}, "feature-idx": 1},
                     {"type": "characters-at", "config": {"pos": [0]}, "feature-idx": 2},
                ],
                [
                    {"type": "characters-at", "config": {"pos": [0]}, "feature-idx": 1},
                    {"type": "characters-at", "config": {"pos": [1]}, "feature-idx": 1},
                ],
                [
                    {"type": "characters-at", "config": {"pos": [0]}, "feature-idx": 2},
                    {"type": "characters-at", "config": {"pos": [1]}, "feature-idx": 2},
                ],
                [
                    {"type": "characters-at", "config": {"pos": [":2"]}, "feature-idx": 3},
                ],
                [
                    {"type": "metaphone", "feature-idx": 1},
                    {"type": "metaphone", "feature-idx": 2},
                ]
            ]
        }
    }

    l_rg_max = []
    l_rg_avg = []
    l_rg_min = []
    l_rr = []
    l_pc = []
    l_qg_max = []
    l_qg_avg = []

    # load data
    file_alice, file_bob = data_sets_pairs[0]
    num_recs_alice = int(file_alice.split('/')[-1].split('_')[0])
    num_recs_bob = int(file_bob.split('/')[-1].split('_')[0])

    print('Loading dataset Alice n={}'.format(num_recs_alice))
    alice_data = pd.read_csv(file_alice)
    alice_data = alice_data.replace(np.nan, '', regex=True)
    alice_data = alice_data.to_dict(orient='split')['data']

    print('Loading dataset Bob n={}'.format(num_recs_bob))
    bob_data = pd.read_csv(file_alice)
    bob_data = bob_data.replace(np.nan, '', regex=True)
    bob_data = bob_data.to_dict(orient='split')['data']
    print('Example data = {}'.format(alice_data[0]))

    for ratio in max_ratio:
        config['config']['filter']['max'] = ratio
        # build candidate blocks
        start_time = time.time()
        print('Building reversed index of Alice')
        block_obj_alice = generate_candidate_blocks(alice_data, config)

        print('Building reversed index of Bob')
        block_obj_bob = generate_candidate_blocks(bob_data, config)
        dbo_time = time.time() - start_time

        # obtain reduction guarantee
        rg_max = 1.0 - block_obj_alice.state.stats['max_size'] / num_recs_alice
        rg_avg = 1.0 - block_obj_alice.state.stats['avg_size'] / num_recs_alice
        rg_min = 1.0 - block_obj_alice.state.stats['min_size'] / num_recs_alice
        print('RG Max={}'.format(rg_max))
        print('RG Avg={}'.format(rg_avg))
        print('RG Min={}'.format(rg_min))

        # obtain quality guarantee
        num_blocks_per_record = block_obj_alice.state.stats['num_of_blocks_per_rec']
        qg_max = max(num_blocks_per_record)
        qg_avg = sum(num_blocks_per_record) / len(num_blocks_per_record)
        print('QG MAX={}'.format(qg_max))
        print('QG AVG={}'.format(qg_avg))

        # build final blocks
        start_time = time.time()
        print('Filtering reversed index - Generate final blocks')
        filtered_alice, filtered_bob = generate_blocks([block_obj_alice, block_obj_bob], K=2)
        lu_time = time.time() - start_time
        num_blocks = len(filtered_alice)

        # assess
        subdata1 = [x[0] for x in alice_data]
        subdata2 = [x[0] for x in bob_data]

        rr, pc = assess_blocks_2party([filtered_alice, filtered_bob], [subdata1, subdata2])

        # combine results together
        l_rg_max.append(rg_max)
        l_rg_min.append(rg_min)
        l_rg_avg.append(rg_avg)
        l_rr.append(rr)
        l_pc.append(pc)
        l_qg_max.append(qg_max)
        l_qg_avg.append(qg_avg)

    df = pd.DataFrame(dict(MAX_RATIO=max_ratio, RG_MAX=l_rg_max, RG_AVG=l_rg_avg, RG_MIN=l_rg_min, RR=l_rr, PC=l_pc,
                           QG_MAX=l_qg_max, QG_AVG=l_qg_avg))
    print(df)
    if mod:
        filename = 'rg_vs_rr_with_mod_n={}.csv'.format(num_recs_alice)
    else:
        filename = 'rg_vs_rr_no_mod_n={}.csv'.format(num_recs_alice)
    df.to_csv(filename, index=False)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        corr_rg_rr(filename)
    else:
        experiment()