Ks = [3]
N = 10000
corruption = 20
rec_id_col = 'recid' if corruption > 0 else 'rec_id'
parties = range(10)
datafile = 'multiparty_dataset/ncvr_numrec_{}_modrec_2_ocp_{}_myp_{}_nump_10.csv'
filenames = [datafile.format(N, corruption, party) for party in parties]
dataname = 'NCVR'

config = {
    # "type": "p-sig",
    # "version": 1,
    # "config": {
    #     "blocking_features": [1, 2, 3],
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
    #             {"type": "characters-at", "config": {"pos": ["0:"]}, "feature-idx": 1},
    #         ],
    #         [
    #             {"type": "characters-at", "config": {"pos": ["0:"]}, "feature-idx": 2},
    #         ],
    #         [
    #             {"type": "characters-at", "config": {"pos": [":2"]}, "feature-idx": 1},
    #             {"type": "characters-at", "config": {"pos": [":2"]}, "feature-idx": 2},
    #
    #         ],
    #         [
    #             {"type": "characters-at", "config": {"pos": ["1:3"]}, "feature-idx": 1},
    #             {"type": "characters-at", "config": {"pos": ["1:3"]}, "feature-idx": 2},
    #         ],
    #         [
    #             {"type": "characters-at", "config": {"pos": ["2:4"]}, "feature-idx": 1},
    #             {"type": "characters-at", "config": {"pos": ["2:4"]}, "feature-idx": 2},
    #         ],
    #         [
    #             {"type": "characters-at", "config": {"pos": ["3:5"]}, "feature-idx": 1},
    #             {"type": "characters-at", "config": {"pos": ["3:5"]}, "feature-idx": 2},
    #         ],
    #         [
    #             {"type": "characters-at", "config": {"pos": [0]}, "feature-idx": 1},
    #             {"type": "characters-at", "config": {"pos": [0]}, "feature-idx": 2},
    #         ],
    #         [
    #             {"type": "characters-at", "config": {"pos": [0]}, "feature-idx": 1},
    #             {"type": "characters-at", "config": {"pos": [1]}, "feature-idx": 2},
    #         ],
    #         [
    #             {"type": "characters-at", "config": {"pos": [0]}, "feature-idx": 1},
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
    "type": "lambda-fold",
    "version": 1,
    "config": {
        "blocking-features": [1, 2],
        "Lambda": 5,
        "bf-len": 2048,
        "num-hash-funcs": 5,
        "K": 40,
        "random_state": 0,
        "input-clks": False
    }
}

