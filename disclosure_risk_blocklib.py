from blocklib import PPRLIndexLambdaFold
from blocklib import generate_blocks, assess_blocks_2party, generate_candidate_blocks
import time
import pandas as pd
from collections import defaultdict
import numpy as np
import pandas as pd

from experiment import match, get_output, get_block_stats, update_result

def disclosure_risk(blocks):
    """Find disclosure risk sorted array back."""
    rec_to_blocks = defaultdict(list)

    for blk_id, records in blocks.items():
        for rec in records:
            rec_to_blocks[rec].append(blk_id)

    # go through rec_to_blocks and calculate risk
    risks = []
    for rec, blks in rec_to_blocks.items():
        if len(blks) == 1:
            try:
                risks.append(1. / len(blocks[blks[0]]))
            except ZeroDivisionError:
                import IPython; IPython.embed()
        else:
            # find intersections
            blk = blocks[blks[0]]
            for b in blks[1:]:
                blk = set(blk).intersection(set(blocks[b]))
                try:
                    risks.append(1. / len(blk))
                except ZeroDivisionError:
                    import IPython;
                    IPython.embed()

    return risks

data_sets_pairs = [
    # ['./datasets/4611_50_overlap_no_mod_alice.csv',
    #  './datasets/4611_50_overlap_no_mod_bob.csv'],
    #
    ['./datasets/46116_50_overlap_no_mod_alice.csv',
     './datasets/46116_50_overlap_no_mod_bob.csv'],

    # ['./datasets/461167_50_overlap_no_mod_alice.csv',
    #  './datasets/461167_50_overlap_no_mod_bob.csv'],

    # ['./datasets/4611676_50_overlap_no_mod_alice.csv',
    #  './datasets/4611676_50_overlap_no_mod_bob.csv'],

]


all_config = {
    'lambda-fold': {
        "type": "lambda-fold",
        "version": 1,
        "config": {
            "blocking-features": [1, 2],
            "Lambda": 5,
            "bf-len": 2000,
            "num-hash-funcs": 10,
            "K": 50,
            "random_state": 0,
            "input-clks": False,
        }
    },
    'p-sig': {
        "type": "p-sig",
        "version": 1,
        "config": {
            "blocking_features": [1],
            # "record-id-col": 0,
            "filter": {
                "type": "count",
                "max": 5000,
                "min": 0.00,
            },
            "blocking-filter": {
                "type": "bloom filter",
                "number-hash-functions": 4,
                "bf-len": 2048,
            },
            "signatureSpecs": [
                [
                     {"type": "characters-at", "config": {"pos": ["0:"]}, "feature-idx": 1},
                ],
                [
                    {"type": "characters-at", "config": {"pos": ["0:"]}, "feature-idx": 2},
                ],
                [
                    {"type": "characters-at", "config": {"pos": [":2"]}, "feature-idx": 1},
                    {"type": "characters-at", "config": {"pos": [":2"]}, "feature-idx": 2},

                ],
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
}

for config in all_config.values():
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

    # build candidate blocks
    start_time = time.time()
    print('Building reversed index of Alice')
    block_obj_alice = generate_candidate_blocks(alice_data, config)

    print('Building reversed index of Bob')
    block_obj_bob = generate_candidate_blocks(bob_data, config)
    dbo_time = time.time() - start_time
    filtered_alice, filtered_bob = generate_blocks([block_obj_alice, block_obj_bob], K=2)
    arisk = disclosure_risk(filtered_alice)
    brisk = disclosure_risk(filtered_bob)
    df = pd.DataFrame(dict(arisk=arisk, brisk=brisk))
    print('Saving to', 'risk_{}.csv'.format(config['type']))
    df.to_csv('risk_{}.csv'.format(config['type']))
