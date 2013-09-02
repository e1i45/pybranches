from collections import defaultdict
import matplotlib.pyplot as pl
import numpy as np
from scipy.interpolate import interp1d

# cpython_history.txt was created with:
# hg log -r 0:tip --template {branch}\t{date}\t{author}\n > cpython_history.txt

fname = '../data/cpython_history.txt'

# "default" branch is replaced with name of upcoming branch
# this is done using the following list of versions.
# Afaik, this does not work reliably for older python versions
# as work was done in the legacy-trunk
branches = ['3.4', '3.3', '3.2', '3.1', '3.0',
            '2.7', '2.6', '2.5', '2.4', '2.3', '2.2', '2.1', '2.0']
branch_next = {k: v for k, v in zip(branches[1:], branches[:-1])}

# TODO: add this information to plot?
'''
Python 2.1 - April 17, 2001
Python 2.2 - December 21, 2001
Python 2.3 - July 29, 2003
Python 2.4 - November 30, 2004
Python 2.5 - September 19, 2006
Python 2.6 - October 1, 2008
Python 2.7 - July 3, 2010
Python 3.0 - December 3, 2008
Python 3.1 - June 27, 2009
Python 3.2 - February 20, 2011
Python 3.3 - September 29, 2012
'''

def time_window_sec(time_window):
    d = 24 * 60 * 60
    if time_window == 'month':
        sec = 4 * 7 * d
    elif time_window == 'week':
        sec = 7 * d
    elif time_window == 'day':
        sec = d
    else:
        raise NotImplementedError(
            'Unexpected value for time_window: {0}.'
            'Try e.g. "month" instead.' .format(time_window))
    return sec


def parse_data_file(fname, time_window='week'):
    group_seconds = time_window_sec(time_window)
    branch_timeunit_authors = defaultdict(lambda: defaultdict(set))
    author_timeunit_branches = defaultdict(lambda: defaultdict(set))
    highest_branch = None
    for line in open(fname):
        parts = line.split('\t')
        if len(parts) == 3:
            branch, date, author = parts
            # date format: unix timestamp
            if '.' in branch:
                if not highest_branch or float(branch) > float(highest_branch):
                    highest_branch = branch
            if branch == 'default':
                branch = branch_next[highest_branch]
            t = int(date.split('.')[0]) // group_seconds
            branch_timeunit_authors[branch][t].add(author)
            author_timeunit_branches[author][t].add(branch)
    return branch_timeunit_authors, author_timeunit_branches


def smooth(xs, w=None):
    if w is None:
        w = [1, 2, 1]  # smoothing weights
    xs2 = [0 for _ in range(len(xs))]
    for i in range(len(xs)):
        min_i = max(i - 2, 0)
        max_i = min(i + 3, len(xs))
        k = w[min_i - i + 2:max_i - i + 2]
        sk = 0
        for k, x in zip(k, xs[min_i:max_i]):
            xs2[i] += k * x
            sk += k
        xs2[i] = xs2[i] * 1.0 / sk
    return xs2


def build_table(branch_timeunit_authors, author_timeunit_branches, time_window):
    group_sec = time_window_sec(time_window)
    ts = set()
    for t_a in branch_timeunit_authors.values():
        ts.update(t_a.keys())
    ts = sorted(ts)  # converts set to list
    rows = []
    branches = sorted(branch_timeunit_authors.keys())
    for branch in branches:
        timeunit_authors = branch_timeunit_authors[branch]
        row = []
        for t in ts:
            # each author only counts as one, if necessary this count is split
            # over several branches (e.g. if author is active in 2 branches,
            # then the author contributes a score of 0.5 per branch)
            score = 0
            for a in timeunit_authors[t]:
                score += 1 / len(author_timeunit_branches[a][t])
            row.append(score)
        row_smoothed = smooth(row)
        rows.append(row_smoothed)
    table = np.array(rows)
    ts = [t * group_sec for t in ts]
    return table, branches, ts


def interp(d):
    f = 2  # new data will be this factor times bigger
    if len(d.shape) == 2:
        n = d.shape[1]
        x = np.linspace(0, n, n)
        x_new = np.linspace(0, n, n * f)
        cs2 = np.ndarray((d.shape[0], n * f))
        for i in range(d.shape[0]):
            f = interp1d(x, d[i, :], kind='cubic')
            cs2[i, :] = f(x_new)
        return cs2
    else:
        n = d.shape[0]
        x = np.linspace(0, n, n)
        x_new = np.linspace(0, n, n * f)
        f = interp1d(x, d, kind='cubic')
        return f(x_new)


def cmap23(idx):
    # increase visual difference between 2 and 3 versions
    p2 = sum(b[0] == '2' for b in branches)
    p3 = sum(b[0] == '3' for b in branches)
    n = len(branches)
    cf = 0.5  # compression factor
    if idx * n < p2:
        idx = idx * cf
    else:
        idx = (idx - p2 / n) * cf + p2 / n + (1 - cf) * p3 / n
    if idx == 1:  # legacy-trunk
        return '#bbbbbb'
    return pl.cm.Spectral(idx)


def plot(tbl, branches, time_units):
    base = [0] * len(tbl[0, :])
    x = time_units
    pl.figure(figsize=(16, 6))
    ax = pl.subplot(111)
    pl.axis('off')
    for i in range(len(branches)):
        c = cmap23(i / (len(branches) - 1))
        idx = np.where(tbl[i, :] > 0.001)
        a = np.min(idx)
        b = np.max(idx) + 1
        y = base[a:b] + tbl[i, a:b]
        y2 = np.hstack((y, base[a:b][::-1]))
        x2 = np.hstack((x[a:b], x[a:b][::-1]))
        ax.fill(x2, y2, facecolor=c, linewidth=1, edgecolor=(0, 0, 0, 0.2))
        base += tbl[i, :]

    maximum = np.max(base)
    year = 0  # 1970
    for year in range(1991, 2014):
        x = (year - 1970) * 365.25 * 24 * 60 * 60
        idx = np.argmin(np.abs(time_units - x))
        ax.vlines(x, maximum / -20, base[idx], colors='#000000', alpha=0.1,
                  linestyles='solid', label=year)
        x2 = 365.25 * 24 * 60 * 60 / 2
        ax.text(x + x2, maximum / -20, year, alpha=0.5,
                horizontalalignment='center')

    pl.gca().set_position([0.0, 0.05, 1.0, 0.89])
    l = pl.legend(branches, loc='lower right',
                  bbox_to_anchor=(0, 0, 1, 1),
                  bbox_transform=pl.gcf().transFigure,
                  ncol=len(branches), labelspacing=0.1, fontsize=12,
                  columnspacing=1)
    l.draw_frame(False)
    pl.gcf().patch.set_facecolor('white')
    pl.show()


def main():
    time_window = 'month'  # week is slow, day is super slow
    (branch_timeunit_authors,
     author_timeunit_branches) = parse_data_file(fname, time_window)
    tbl, branches, time_units = build_table(branch_timeunit_authors,
                                            author_timeunit_branches,
                                            time_window)

    # optional, but makes it look a bit smoother
    tbl = interp(tbl)
    time_units = interp(np.array(time_units))

    plot(tbl, branches, time_units)


if __name__ == '__main__':
    main()
