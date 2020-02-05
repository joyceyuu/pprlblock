import pandas as pd
import matplotlib.pyplot as pl

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



methods = ['hlsh', 'snc-3p', 'k-nn', 'lambda-fold', 'p3-sig']

for method in methods:
    df = pd.read_csv('risk_{}.csv'.format(method))
    color = colors[method.lower()]
    if 'lambda' in method.lower():
        method = r'$\Lambda$' + '-LSH'

    pl.plot(sorted(list(df['arisk'].values)), label=method, color=color)

pl.grid()
pl.xlabel('Number of records')
pl.ylabel('Sorted Disclosure Risk')
pl.title('Sorted Disclosure Risk')
pl.legend()
pl.savefig('figures/Disclosure_Risk_4k.eps')