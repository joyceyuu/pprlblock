import matplotlib.pyplot as pl
import numpy as np
import sys
import pandas as pd
from vis import get_colors_and_shapes

colors = {
    'p3-sig': 'tab:pink',
    'lambda-fold': 'tab:olive',
    'k-nn': 'tab:cyan',
    'snc-3p': 'tab:blue',
    # 'snc-3p-size': 'tab:orange',
    'hlsh': 'tab:green',
    'snc-2p': 'tab:purple',
    'hclust': 'tab:brown',
          }

def draw_blocks(filenames):
    pl.figure(figsize=(8, 6))

    padding = [0, 0.2]
    for filename, pad in zip(filenames, padding):
        method = filename.split('.csv')[0].split('_')[-1].upper()
        color = colors[method.lower()]
        if 'lambda' in method.lower():
            method = r'$\Lambda$' + '-LSH'
        psig = pd.read_csv(filename)
        avgs = psig['avg_blk'].values
        mins = psig['min_blk'].values
        maxs = psig['max_blk'].values
        stds = psig['std_dev'].values
        parties = range(len(psig))
        pl.errorbar([x + pad for x in parties], avgs, stds, fmt='ok', lw=3, label=method, ecolor=color)
        pl.errorbar([x + pad for x in parties], avgs, [avgs - mins, maxs - avgs], fmt='.k', ecolor='gray', lw=1)
    pl.xticks(parties, ['Party_{}'.format(x + 1) for x in parties])
    pl.legend()
    outfile = 'figures/ABS_Block_Size_Distribution.eps'.format(method)
    pl.yscale('log')

    pl.title('ABS Employee Data Block Size Distribution')
    pl.savefig(outfile)
    # pl.show()


if __name__ == '__main__':
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    draw_blocks([filename1, filename2])