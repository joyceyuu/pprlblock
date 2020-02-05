import pandas as pd
import matplotlib.pyplot as pl

scales = ['4611', '46116', '461167']
sizes = [4, 40, 400]
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
runtime = {k: [] for k in COLORS}
for s in scales:
    df = pd.read_csv(filename.format(s))
    for name, t1, t2 in zip(df['Method'], df['dbo_time'], df['lu_time']):
        runtime[name].append(t1 + t2)

for name, times in runtime.items():
    color = COLORS[name]
    name = name.upper()
    if 'lambda' in name.lower():
        method = r'$\Lambda$' + '-LSH'
    pl.plot(range(len(times)), times, 'o-', color=color, label=name)


pl.legend()
pl.xticks(range(3), ['4k', '40k', '400k'])
pl.grid()
pl.yscale('log')
pl.xlabel('Scale')
pl.ylabel('Total Running Time')
pl.title('Total Running Time versus Scale of Dataset')
pl.savefig('figures/Scalability.eps')
# pl.show()

