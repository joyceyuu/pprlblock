from blocklib import PPRLIndexLambdaFold
from blocklib import generate_blocks, assess_blocks_2party, generate_candidate_blocks
import time
import pandas as pd
import numpy as np
import pandas as pd

from experiment import match, get_output, get_block_stats, update_result


data_sets_pairs = [
    ['./datasets/4611_50_overlap_no_mod_alice.csv',
     './datasets/4611_50_overlap_no_mod_bob.csv'],

    # ['./datasets/46116_50_overlap_no_mod_alice.csv',
    #  './datasets/46116_50_overlap_no_mod_bob.csv'],

    # ['./datasets/461167_50_overlap_no_mod_alice.csv',
    #  './datasets/461167_50_overlap_no_mod_bob.csv'],
    #
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
            "num-hash-funcs": 5,
            "K": 50,
            "random_state": 0,
            "input-clks": False,
        }
    },
    # 'p-sig': {
    #     "type": "p-sig",
    #     "version": 1,
    #     "config": {
    #         "blocking_features": [1],
    #         # "record-id-col": 0,
    #         "filter": {
    #             "type": "ratio",
    #             "max": 0.02,
    #             "min": 0.00,
    #         },
    #         "blocking-filter": {
    #             "type": "bloom filter",
    #             "number-hash-functions": 4,
    #             "bf-len": 2048,
    #         },
    #         "signatureSpecs": [
    #             [
    #                  {"type": "characters-at", "config": {"pos": [0]}, "feature-idx": 1},
    #                  {"type": "characters-at", "config": {"pos": [0]}, "feature-idx": 2},
    #             ],
    #             [
    #                 {"type": "characters-at", "config": {"pos": [0]}, "feature-idx": 1},
    #                 {"type": "characters-at", "config": {"pos": [1]}, "feature-idx": 1},
    #             ],
    #             [
    #                 {"type": "characters-at", "config": {"pos": [0]}, "feature-idx": 2},
    #                 {"type": "characters-at", "config": {"pos": [1]}, "feature-idx": 2},
    #             ],
    #             [
    #                 {"type": "characters-at", "config": {"pos": [":2"]}, "feature-idx": 3},
    #             ],
    #             [
    #                 {"type": "metaphone", "feature-idx": 1},
    #                 {"type": "metaphone", "feature-idx": 2},
    #             ]
    #         ]
    #     }
    # }
}

for config in all_config.values():
    # load data
    file_alice, file_bob = data_sets_pairs[0]
    dbo_time, lu_time, num_blocks, rr, pc, num_recs_alice, block_obj_alice, block_obj_bob = match(file_alice, file_bob, config)
    block_alice_stats = get_block_stats(block_obj_alice)
    block_bob_stats = get_block_stats(block_obj_bob)

    # get result dataframe
    method_name = config['type']
    df = get_output(method_name, num_recs_alice, num_recs_alice, None, 3, rr, pc, None, dbo_time, lu_time, None,
                    block_alice_stats, block_bob_stats, num_blocks, 100)
    print(df.T)

    # update result
    # update_result(df, method_name, 'result2')