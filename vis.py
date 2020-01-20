import matplotlib.pyplot as pl
import pandas as pd
import numpy as np
import sys

COLORS = {
    'k-nn': 'tab:cyan',
    'snc-3p': 'tab:blue',
    # 'snc-3p-size': 'tab:orange',
    'hlsh': 'tab:green',
    'snc-2p': 'tab:purple',
    'hclust': 'tab:brown',
    'p3-sig': 'tab:pink',
    'lambda-LSH': 'tab:olive',
}

SHAPES = {
    'k-nn': 'o',
    'snc-3p': 'v',
    # 'snc-3p-size': 's',
    'hlsh': 'D',
    'snc-2p': '*',
    'hclust': '8',
    'p3-sig': 'X',
    'lambda-LSH': 'p',
}

def get_colors_and_shapes(res):
    methods = res['Method'].values
    shapes = [SHAPES[x] for x in methods]
    colors = [COLORS[x] for x in methods]
    methods = res['Method'].values
    methods = [x.upper() for x in methods]
    for i in range(len(methods)):
        if 'lambda' in methods[i].lower():
            methods[i] = r'$\Lambda$' + '-LSH'
    print(methods)
    return methods, shapes, colors


def blocksize(filename):
    """Draw blocksize as errorbar to see distribution."""
    res = pd.read_csv(filename)
    # res = res[res['Method'].map(lambda x: x != 'hclust')]
    n = len(res)
    nm = ['{}_min_blk', '{}_med_blk', '{}_max_blk', '{}_avg_blk', '{}_std_dev']
    alice = [x.format('a') for x in nm]
    bob = [x.format('b') for x in nm]
    pl.figure(figsize=(6, 6))
    pl.subplot(2, 2, 1)
    draw_errorbar(n, res, alice, 'Alice')
    pl.subplot(2, 2, 2)
    draw_errorbar(n, res, bob, 'Bob')
    pl.subplot(2, 2, 3)
    draw_ratios(res)
    pl.subplot(2, 2, 4)
    draw_time(res)
    pl.show()

def draw_errorbar(n, res, nm, partyname):
    """Draw errorbar."""
    mins = res[nm[0]].values
    meds = res[nm[1]].values
    maxs = res[nm[2]].values
    avgs = res[nm[3]].values
    stds = res[nm[4]].values
    pl.errorbar(np.arange(n), avgs, stds, fmt='ok', lw=3)
    pl.errorbar(np.arange(n), avgs, [avgs - mins, maxs - avgs],
                fmt='.k', ecolor='gray', lw=1)
    methods, shapes, colors = get_colors_and_shapes(res)
    pl.xticks(np.arange(n), methods)
    pl.yscale('log')
    pl.title('Block Size Distribution of ' + partyname)


def draw_ratios(res):
    """Draw different ratios."""
    n = len(res)
    rr = res['rr'].values
    pc = res['pc'].values
    pq = res['pq'].values
    methods, shapes, colors = get_colors_and_shapes(res)
    for x1, x2, name, marker, color in zip(rr, pc, methods, shapes, colors):
        pl.plot([x1], [x2], marker=marker, linestyle='', ms=6, label=name, alpha=0.8, color=color)
        pl.text(x1, x2, name, fontsize=8)
    pl.xlabel('Reduction Ratio')
    pl.ylabel('Pair Completeness')
    pl.legend()
    # pl.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
    #           fancybox=True, ncol=n)
    pl.grid()
    pl.title('Reduction ratio versus Pair completeness')


def draw_time(res):
    """Draw running time."""
    n = len(res)
    dob_time = res['dbo_time'].values
    lu_time = res['lu_time'].values
    methods, shapes, colors = get_colors_and_shapes(res)
    nrange = np.arange(n)
    for i, dtime, ltime, marker, name, color in zip(nrange, dob_time, lu_time, shapes, methods, colors):
        tot_time = dtime + ltime
        pl.plot(i, tot_time, marker=marker, label=name, color=color)
        pl.text(i, tot_time, name, fontsize=8)
    pl.xticks(np.arange(n), methods)
    pl.title('Total Running Time (Log-Scale)')
    pl.yscale('log')
    pl.ylabel('Total Running Time (log-scale)')
    pl.grid()

def draw_risk(arisk, brisk, method, n):
    """Draw disclosure risk."""
    pl.figure(figsize=(12, 12))
    pl.plot(sorted(arisk.values()), label='Alice')
    pl.plot(sorted(brisk.values()), label='Bob')
    pl.legend()
    pl.grid()
    pl.ylabel('Disclosure risk')
    pl.title('Sorted Disclosure Risk of {} n={}'.format(method, n))
    pl.savefig('{}_{}.pdf'.format(method, n))

def draw_riskcompare(risks, methods, dname, n):
    """Draw disclosure risk."""
    pl.figure(figsize=(12, 12))
    for risk, method in zip(risks, methods):
        pl.plot(sorted(risk.values()), label=method)
    pl.legend()
    pl.grid()
    pl.ylabel('Disclosure risk')
    pl.title('Sorted Disclosure Risk of {} n={}'.format(dname, n))
    pl.savefig('{}_{}.pdf'.format(dname, n))


def draw_drop_ratio(fname):
    """Draw reduction ratio and pair completeness for different drop ratio."""
    res = pd.read_csv(fname)
    x = res['drop_ratio'].values
    rr = res['rr'].values
    pc = res['pc'].values
    pl.figure(figsize=(8, 6))
    pl.plot(x, rr, label='reduction ratio')
    pl.plot(x, pc, label='pair completeness')
    pl.title('Reduction Ratio and Pair Completeness versus Drop Ratio (n={})'
             .format(res['alice_num_recs'].unique()[0]))
    pl.grid()
    pl.legend()
    pl.show()


def corr_rg_rr(fname):
    """Draw reduction ratio versus reduction guarantee correlation graph."""
    df = pd.read_csv(fname)
    rg_max = df['RG_MAX'].values
    rg_min = df['RG_MIN'].values
    rg_avg = df['RG_AVG'].values
    size = fname.split('.csv')[0].split('n=')[-1]
    rr = df['RR'].values
    pl.figure(figsize=(8, 6))
    pl.scatter(rg_max, rr, label='RG_MAX')
    pl.grid()
    pl.legend()
    pl.title('Reduction Guarantee versus Reduction Ratio')
    pl.xlabel('Reduction Guarantee')
    pl.ylabel('Reduction Ratio')
    pl.savefig('figures/corr_rr_rg_4611_nomod.eps')

    qg_avg = df['QG_AVG'].values
    pc = df['PC'].values
    pl.figure(figsize=(8, 6))
    pl.scatter(qg_avg, pc, label='QG_AVG')
    pl.grid()
    pl.legend()
    pl.title('Quality Guarantee versus Pair Completeness')
    pl.xlabel('Quality Guarantee')
    pl.ylabel('Pair Completeness')
    pl.savefig('figures/corr_pc_qg_{}_nomod.eps'.format(size))

if __name__ == '__main__':
    filename = sys.argv[1]
    # blocksize(filename)

    res = pd.read_csv(filename)
    res = res[res['Method'].map(lambda x: x != 'hclust')]
    n = int(filename.split('=')[-1].split('.csv')[0])
    print(n)
    nm = ['{}_min_blk', '{}_med_blk', '{}_max_blk', '{}_avg_blk', '{}_std_dev']
    alice = [x.format('a') for x in nm]
    bob = [x.format('b') for x in nm]
    pl.figure()
    draw_errorbar(len(res), res, alice, 'Alice')
    pl.tight_layout()
    pl.savefig('figures/Block_Size_Boxplot_Alice_{}.eps'.format(n))

    # pl.subplot(2, 2, 2)

    pl.figure(figsize=(6, 6))
    draw_errorbar(len(res), res, bob, 'Bob')
    pl.savefig('figures/Block_Size_Boxplot_Bob_{}.eps'.format(n))

    # pl.subplot(2, 2, 3)
    pl.figure()
    draw_ratios(res)
    pl.savefig('figures/RR_versus_PC_{}.eps'.format(n))

    # pl.subplot(2, 2, 4)
    pl.figure()
    draw_time(res)
    pl.tight_layout()
    pl.savefig('figures/Total_Running_Time_{}.eps'.format(n))
