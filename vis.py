import matplotlib.pyplot as pl
import pandas as pd
import numpy as np
import sys

def blocksize(filename):
    """Draw blocksize as errorbar to see distribution."""
    res = pd.read_csv(filename)
    res = res[res['Method'].map(lambda x: x != 'hclust')]
    n = len(res)
    nm = ['{}_min_blk', '{}_med_blk', '{}_max_blk', '{}_avg_blk', '{}_std_dev']
    alice = [x.format('a') for x in nm]
    bob = [x.format('b') for x in nm]
    pl.figure(figsize=(12, 12))
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
    methods = res['Method'].values
    pl.xticks(np.arange(n), methods)
    pl.title('Block Size Distribution of ' + partyname)


def draw_ratios(res):
    """Draw different ratios."""
    n = len(res)
    rr = res['rr'].values
    pc = res['pc'].values
    pq = res['pq'].values
    methods = res['Method'].values
    shapes = ['o', 'v', 's', 'D', '*', '8']
    for x1, x2, name, marker in zip(rr, pc, methods, shapes):
        pl.plot([x1], [x2], marker=marker, linestyle='', ms=6, label=name, alpha=0.8)
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
    methods = res['Method'].values
    shapes = ['o', 'v', 's', 'D', '*', '8']
    nrange = np.arange(n)
    for i, dtime, ltime, marker, name in zip(nrange, dob_time, lu_time, shapes, methods):
        tot_time = dtime + ltime
        pl.plot(i, tot_time, marker=marker, label=name)
        pl.text(i, tot_time, name, fontsize=8)
    methods = res['Method'].values
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
             .format(46116))
    pl.grid()
    pl.legend()
    pl.show()


if __name__ == '__main__':
    filename = sys.argv[1]
    blocksize(filename)
