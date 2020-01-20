
datafile = 'abs_employee/ABS_Employee_{}.csv'
parties = ['A', 'B', 'C']
Ks = [3]
filenames = [datafile.format(p) for p in parties]
rec_id_col = 'ENTID'
dataname = 'ABS'

config = {
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
    #             {"type": "characters-at", "config": {"pos": ["0:"]}, "feature-idx": 1},
    #         ],
    #         [
    #             {"type": "characters-at", "config": {"pos": ["0:"]}, "feature-idx": 2},
    #         ],
    #         [
    #             {"type": "characters-at", "config": {"pos": ["0:"]}, "feature-idx": 3},
    #         ],
    #         [
    #             {"type": "characters-at", "config": {"pos": ["0:"]}, "feature-idx": 4},
    #         ],
    #         [
    #             {"type": "characters-at", "config": {"pos": ["0:"]}, "feature-idx": 5},
    #         ],
    #     ]
    # }

    "type": "lambda-fold",
    "version": 1,
    "config": {
        "blocking-features": [1, 2, 3, 4, 5],
        "Lambda": 5,
        "bf-len": 2048,
        "num-hash-funcs": 5,
        "K": 40,
        "random_state": 0,
        "input-clks": False
    }
}


