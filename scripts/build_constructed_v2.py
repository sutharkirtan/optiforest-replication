"""Constructed benchmark v2: typed, graded anomalies injected into two clean bases.
Usage (repo root): python3 scripts/build_constructed_v2.py path/to/creditcard.csv
Bases : AD    = UCI Internet Advertisements, normal class (2,820 x 1,555 binary, sparse)
        CC    = ULB Credit Card Fraud, normal class, 2,820-row random sample, 30 features standardised
Types : global  - a few features pushed far from normal (easy)
        local   - many features moved slightly / one block of features swapped (hard: stays inside the data envelope)
        cluster - a tight group of near-identical anomalies (collective anomaly)
        shuffle - features drawn from different rows: marginals kept, correlations destroyed
Severity 1 (subtle) .. 3 (strong). Contamination 5%. Seed 42.
Outputs: data/v2/<base>_<type>_s<sev>.csv (headerless, label last) and data/v2/manifest.csv (the annotation: file, row, is_anomaly, type, severity).
"""
import sys, numpy as np, pandas as pd
SEED, RATE = 42, 0.05
rng0 = np.random.default_rng(SEED)
ad = pd.read_csv('data/ad.csv', header=None).values
A = ad[ad[:, -1] == 0][:, :-1].astype(np.int8)
cc = pd.read_csv(sys.argv[1]); ccn = cc[cc.Class == 0].drop(columns='Class').sample(len(A), random_state=SEED)
B = ((ccn - ccn.mean()) / ccn.std()).values.astype(np.float64)
SEV = {1: 0, 2: 1, 3: 2}

def shuffle_cols(base, k, frac, rng):
    n, m = base.shape; rows = base[rng.integers(0, n, k)].copy()
    cols = rng.choice(m, int(round(frac * m)), replace=False)
    rows[:, cols] = base[rng.integers(0, n, (k, len(cols))), cols]
    return rows

def make_ad(t, s, k, rng):
    n, m = A.shape; i = SEV[s]
    if t == 'global':      # flip 0.5 / 1 / 2 % of bits
        r = A[rng.integers(0, n, k)].copy(); f = int(round([0.005, 0.01, 0.02][i] * m))
        for x in r: j = rng.choice(m, f, replace=False); x[j] = 1 - x[j]
        return r
    if t == 'local':       # mosaic: swap a block of 5/15/30 % of columns with another normal row's block
        r = A[rng.integers(0, n, k)].copy(); w = int(round([0.05, 0.15, 0.30][i] * m))
        for x in r: st = rng.integers(0, m - w); x[st:st + w] = A[rng.integers(0, n), st:st + w]
        return r
    if t == 'cluster':     # one prototype (2% flips) copied with 0.1/0.2/0.5 % extra flips
        p = A[rng.integers(0, n)].copy(); j = rng.choice(m, int(0.02 * m), replace=False); p[j] = 1 - p[j]
        r = np.tile(p, (k, 1)); f = max(1, int(round([0.005, 0.002, 0.001][i] * m)))
        for x in r: j = rng.choice(m, f, replace=False); x[j] = 1 - x[j]
        return r
    return shuffle_cols(A, k, [0.3, 0.6, 1.0][i], rng)

def make_cc(t, s, k, rng):
    n, m = B.shape; i = SEV[s]
    if t == 'global':      # 5 features shifted by +-2 / 3.5 / 5 sigma
        r = B[rng.integers(0, n, k)].copy()
        for x in r: j = rng.choice(m, 5, replace=False); x[j] += rng.choice([-1, 1], 5) * [2.0, 3.5, 5.0][i]
        return r
    if t == 'local':       # 10 features moved by +-0.8 / 1.5 / 2.5 sigma
        r = B[rng.integers(0, n, k)].copy()
        for x in r: j = rng.choice(m, 10, replace=False); x[j] += rng.choice([-1, 1], 10) * [0.8, 1.5, 2.5][i]
        return r
    if t == 'cluster':     # tight Gaussian blob offset by 1.5 / 2.5 / 4 sigma on 10 features
        c = B[rng.integers(0, n)].copy(); j = rng.choice(m, 10, replace=False); c[j] += rng.choice([-1, 1], 10) * [1.5, 2.5, 4.0][i]
        return c + rng.normal(0, 0.1, (k, m))
    return shuffle_cols(B, k, [0.3, 0.6, 1.0][i], rng)

rows = []
for base, X, mk in (('ad', A, make_ad), ('cc', B, make_cc)):
    n = len(X); k = int(round(n * RATE / (1 - RATE)))
    for t in ('global', 'local', 'cluster', 'shuffle'):
        for s in (1, 2, 3):
            rng = np.random.default_rng(SEED + 1000 * s + 100 * 'global local cluster shuffle'.split().index(t) + (0 if base == 'ad' else 7))
            an = mk(t, s, k, rng); data = np.vstack([X, an]); y = np.r_[np.zeros(n), np.ones(len(an))]
            name = f'{base}_{t}_s{s}'
            out = np.column_stack([data, y])
            if base == 'ad': out = out.astype(int)   # binary base: write integers, keeps files small
            pd.DataFrame(out).to_csv(f'data/v2/{name}.csv', header=False, index=False, **({} if base == 'ad' else {'float_format': '%.4f'}))
            rows += [(name, r, int(y[r]), 'normal' if y[r] == 0 else t, 0 if y[r] == 0 else s) for r in range(len(y))]
            print(name, data.shape, 'anomalies', len(an), flush=True)
pd.DataFrame(rows, columns=['file', 'row', 'is_anomaly', 'type', 'severity']).to_csv('data/v2/manifest.csv', index=False)
