import matplotlib.pyplot as pl
import numpy as np
import sys
import pandas as pd
from vis import get_colors_and_shapes

COLORS = {
    'p3-sig': 'tab:pink',
    'lambda-LSH': 'tab:olive',
    'k-nn': 'tab:cyan',
    'snc-3p': 'tab:blue',
    # 'snc-3p-size': 'tab:orange',
    'hlsh': 'tab:green',
    'snc-2p': 'tab:purple',
    'hclust': 'tab:brown',
          }


def get_colors_and_shapes(res):
    methods = res['Method'].values
    colors = [COLORS[x] for x in methods]
    methods = res['Method'].values
    methods = [x.upper() for x in methods]
    for i in range(len(methods)):
        if 'lambda' in methods[i].lower():
            methods[i] = r'$\Lambda$' + '-LSH'
    print(methods)
    return methods, colors


def draw_blocks(filename):
    pl.figure(figsize=(8, 6))

    padding = [0, 0.2]

    parties = ['a', 'b']
    df = pd.read_csv(filename)
    methods, colors = get_colors_and_shapes(df)
    ecolors = ['tab:orange', 'tab:blue']
    ns = range(len(methods))
    for party, pad, color in zip(parties, padding, ecolors):
        avgs = df[party + '_' + 'avg_blk'].values
        mins = df[party + '_' + 'min_blk'].values
        maxs = df[party + '_' + 'max_blk'].values
        stds = df[party + '_' + 'std_dev'].values
        pl.errorbar([x + pad for x in ns], avgs, stds, fmt='ok', lw=3, label='Party_{}'.format(party.upper()), ecolor=color)
        pl.errorbar([x + pad for x in ns], avgs, [avgs - mins, maxs - avgs], fmt='.k', ecolor='gray', lw=1)
    pl.xticks(ns, methods)
    pl.legend()
    outfile = 'figures/NCVR_Block_Size_Distribution.eps'
    pl.yscale('log')

    pl.title('Block Size Distribution')
    pl.savefig(outfile)
    # pl.show()


if __name__ == '__main__':
    filename1 = sys.argv[1]
    draw_blocks(filename1)