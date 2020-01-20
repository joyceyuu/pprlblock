import sys
import math
import time
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from blocklib import generate_candidate_blocks, generate_blocks
import config_ncvr as config
from experiment import get_block_stats
import config_abs as config


def generate_candidates(filenames, config, parties, rec_id_col):
    """Generate candidates for each party."""
    block_objs = []
    truth = []
    data = []

    for party, filename in zip(parties, filenames):
        print('Loading file {} and Generate candidate blocks'.format(filename))
        df = pd.read_csv(filename).astype(str)
        records = df.to_dict(orient='split')['data']
        data.append(df[rec_id_col].values)

        # append record ids
        truth.append(pd.DataFrame({'id{}'.format(party): df.index, rec_id_col: df[rec_id_col]}))
        print('Loaded {} records from file {}'.format(len(df), filename))
        # generate candidate blocks
        blk_obj = generate_candidate_blocks(records, config)
        block_objs.append(blk_obj)

    return block_objs, data, truth


def reduction_ratio(filtered_reverse_indices, data, K):
    """Assess reduction ratio for multiple parties."""
    naive_num_comparison = 1
    for d in data:
        naive_num_comparison *= len(d)

    block_keys = defaultdict(int)  # type: Dict[Any, int]
    for reversed_index in filtered_reverse_indices:
        for key in reversed_index:
            block_keys[key] += 1
    final_block_keys = [key for key, count in block_keys.items() if count >= K]

    reduced_num_comparison = 0
    for key in final_block_keys:
        num_comparison = 1
        for reversed_index in filtered_reverse_indices:
            index = reversed_index.get(key, [0])
            num_comparison *= len(index)
        reduced_num_comparison += num_comparison
    rr = 1 - reduced_num_comparison / naive_num_comparison
    return rr, reduced_num_comparison, naive_num_comparison


def set_completeness(filtered_reverse_indices, truth, K):
    """Assess reduction ratio for multiple parties."""
    block_keys = defaultdict(int)  # type: Dict[Any, int]
    for reversed_index in filtered_reverse_indices:
        for key in reversed_index:
            block_keys[key] += 1
    final_block_keys = [key for key, count in block_keys.items() if count >= K]

    sets = defaultdict(set)
    for i, reversed_index in enumerate(filtered_reverse_indices):
        for key in final_block_keys:
            index = reversed_index.get(key, None)
            if index is not None:
                for ind in index:
                    sets[key].add((i, ind))

    num_true_matches = 0
    for true_set in tqdm(truth):
        check = False
        true_set = set(true_set)
        for s in sets.values():
            if true_set.intersection(s) == true_set:
                check = True
        if check:
            num_true_matches += 1

    sc = num_true_matches / len(truth)
    return sc


def experiment(block_objs, data, k, method):
    """Run multiparty matching."""
    # generate final blocks
    print('Generating final blocks...')
    filtered_reversed_indices = generate_blocks(block_objs, k)
    total_time = time.time() - start_time
    save_blocks(filtered_reversed_indices, data, method)
    # get ground truth
    print('Getting ground truth')
    df_truth = truth[0].merge(truth[1], on=config.rec_id_col, how='outer')
    for df in truth[2:]:
        df_truth = df_truth.merge(df, on=config.rec_id_col, how='outer')
    df_truth = df_truth.drop(columns=[config.rec_id_col])

    true_matches = set()
    for row in df_truth.itertuples(index=False):
        cand = [(i, int(x)) for i, x in enumerate(row) if not math.isnan(x)]
        if len(cand) >=k:
            true_matches.add(tuple(cand))
    print(f'we have {len(true_matches)} true matches')
    # evaluate
    print('Calculating reduction ratio...')
    rr, reduced_num_comparison, _ = reduction_ratio(filtered_reversed_indices, data, k)
    print('Calculating set completeness...')
    sc = set_completeness(filtered_reversed_indices, true_matches, k)

    print('rr={}'.format(rr))
    print('sc={}'.format(sc))

    return rr, sc, reduced_num_comparison, total_time


def save_blocks(filtered_reversed_indices, data, method):
    """For each party, save blocks to csv."""
    for i, reversed_index in enumerate(filtered_reversed_indices):
        entities = data[i]
        filename = '{}_Blocks_Party_{}_{}.csv'.format(config.dataname, i, method)
        print('Saving to', filename)
        with open(filename, 'w') as f:
            for blk_key, records in reversed_index.items():
                rids = [str(entities[ind]) for ind in records]
                if type(blk_key) == tuple:
                    line = [''.join([str(x) for x in blk_key])] + rids
                else:
                    line = [blk_key] + rids
                f.write(','.join(line))
                f.write('\n')


import matplotlib.pyplot as pl
import numpy as np
from vis import get_colors_and_shapes


def subset_matching(filename):
    df = pd.read_csv(filename)
    k = range(2, 10)
    rr = df['RR'].values
    sc = df['SC'].values
    runtime = df['TOTAL_TIME'].values
    n = filename.split('.csv')[0].split('n=')[-1]
    pl.figure()
    pl.plot(k, rr)
    pl.xlabel(r'$s_m$')
    pl.ylabel('Reduction Ratio')
    pl.title('Reduction Ratio versus Minimum Subset Size')
    pl.savefig('figures/subset_matching_sm_rr_n=10k.eps')

    pl.figure()
    pl.plot(k, sc)
    pl.xlabel(r'$s_m$')
    pl.ylabel('Set Completeness')
    pl.title('Set Completeness versus Minimum Subset Size')
    pl.savefig('figures/subset_matching_sm_sc_n=10k.eps')

    pl.figure()
    pl.plot(k, runtime)
    pl.xlabel(r'$s_m$')
    pl.ylabel('Running Time')
    pl.title('Running Time versus Minimum Subset Size')
    pl.savefig('figures/subset_matching_sm_runtime_n=10k.eps'.format(n))


def draw_ratios(res):
    """Draw different ratios."""
    rr = res['RR'].values
    pc = res['SC'].values
    methods, shapes, colors = get_colors_and_shapes(res)
    for x1, x2, name, marker, color in zip(rr, pc, methods, shapes, colors):
        pl.plot([x1], [x2], marker=marker, linestyle='', ms=6, label=name, alpha=0.8, color=color)
        pl.text(x1, x2, name, fontsize=8)
    pl.xlabel('Reduction Ratio')
    pl.ylabel('Set Completeness')
    pl.legend()
    # pl.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
    #           fancybox=True, ncol=n)
    pl.grid()
    pl.title('Reduction ratio versus Set completeness')


def draw_time(res):
    """Draw running time."""
    n = len(res)
    total_time = res['TOTAL_TIME'].values
    methods, shapes, colors = get_colors_and_shapes(res)
    nrange = np.arange(n)
    for i, tot_time, marker, name, color in zip(nrange, total_time, shapes, methods, colors):
        pl.plot(i, tot_time, marker=marker, label=name, color=color)
        pl.text(i, tot_time, name, fontsize=8)
    pl.xticks(np.arange(n), methods)
    pl.title('Total Running Time (Log-Scale)')
    pl.yscale('log')
    pl.ylabel('Total Running Time (log-scale)')
    pl.grid()

if __name__ == '__main__':

    if len(sys.argv) > 1 and sys.argv[1] == 'sm':
        subset_matching(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == 'ratio':
        res = pd.read_csv(sys.argv[2])
        pl.figure()
        draw_ratios(res)
        pl.savefig('figures/RR_versus_SC_ABS.eps')

        pl.figure()
        draw_time(res)
        pl.savefig('figures/Total_Running_Time_ABS.eps')

    else:
        rrs, scs, tots = [], [], []
        num_comparisons = []
        start_time = time.time()
        block_objs, data, truth = generate_candidates(config.filenames, config.config, config.parties,
                                                      config.rec_id_col)
        # get block stats
        stats = []
        stats_names = ['min_blk', 'med_blk', 'max_blk', 'avg_blk', 'std_dev']
        for obj in block_objs:
            o_stats = get_block_stats(obj)
            stats.append(o_stats)

        df_stats = pd.DataFrame(data=stats, columns=stats_names)
        filename = 'multiparty_blk_stats_{}_{}.csv'.format(config.dataname, config.config['type'])
        print('Saving to', filename)
        df_stats.to_csv(filename, index=False)

        N = max([len(d) for d in data])
        for k in config.Ks:
            print(f'\n\nSubset matching with k={k}')
            print('==============================================================================')

            rr, sc, reduced_num_comparison, total_time = experiment(block_objs, data, k, config.config['type'])

            rrs.append(rr)
            scs.append(sc)
            tots.append(total_time)
            num_comparisons.append(reduced_num_comparison)

        df = pd.DataFrame(dict(k=k, NUMBER_OF_COMPARISON=num_comparisons, RR=rrs, SC=scs, TOTAL_TIME=total_time))
        df.to_csv('multiparty_{}_{}_n={}.csv'.format(config.dataname, config.config['type'], N), index=False)
        print(df)