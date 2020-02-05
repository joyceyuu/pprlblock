import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

scales = ['4611', '46116', '461167']

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

filename = 'result2_n={}.csv'
pcs = {k: [] for k in COLORS}
rrs = {k: [] for k in COLORS}
for s in scales:
    df = pd.read_csv(filename.format(s))
    for name, pc, rr in zip(df['Method'], df['pc'], df['rr']):
        pcs[name].append(pc)
        rrs[name].append(rr)


for i, (name, nums) in enumerate(pcs.items()):
    color = COLORS[name]
    mean = np.mean(nums)
    std = np.std(nums)
    name = name.upper()
    if 'lambda' in name.lower():
        method = r'$\Lambda$' + '-LSH'
    pl.plot(i, mean, 'o', color=color, label=name)
    pl.plot(i, mean - std, 'o', color=color)
    pl.plot(i, mean + std, 'o', color=color)
    pl.vlines(i, mean - std, mean + std, color=color)

names = []
for name in pcs:
    name = name.upper()
    if 'lambda' in name.lower():
        method = r'$\Lambda$' + '-LSH'
    names.append(name)
pl.legend()
pl.grid()
pl.xlabel('Method')
pl.ylabel('Pair Completeness')
pl.title('Pair Completeness Variation Versus Methods')
pl.xticks(range(len(pcs)), names)
pl.savefig('figures/PC_4k_variation.eps')

