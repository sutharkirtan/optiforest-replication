"""Build the constructed anomaly-injection datasets.
Clean base = normal class of UCI Internet Advertisements (ad.csv, label 0, 2820 rows x 1555 binary features).
Anomalies are synthetic (labelled 1) and of two types, 50/50:
  flip   : copy a normal row, flip 2% of its bits (feature noise)
  shuffle: build a row by taking each feature from a different random normal row (marginals kept, correlations destroyed)
Contamination rates 1,5,10,20%. Fixed seed. Output: headerless CSV, label in last column (same format as ad.csv).
"""
import numpy as np, pandas as pd
SEED = 42
d = pd.read_csv('data/ad.csv', header=None).values
normal = d[d[:, -1] == 0][:, :-1].astype(np.int8)
n, m = normal.shape
for rate in (0.01, 0.05, 0.10, 0.20):
    rng = np.random.default_rng(SEED)
    k = int(round(n * rate / (1 - rate)))          # anomalies so that k/(n+k)=rate
    kf = k // 2; ks = k - kf
    flip = normal[rng.integers(0, n, kf)].copy()
    for r in flip:
        idx = rng.choice(m, int(0.02 * m), replace=False)
        r[idx] = 1 - r[idx]
    shuf = normal[rng.integers(0, n, (ks, m)), np.arange(m)]
    X = np.vstack([normal, flip, shuf])
    y = np.r_[np.zeros(n), np.ones(kf + ks)]
    out = np.column_stack([X, y]).astype(int)
    name = f'data/constructed/ad_inj{int(rate*100):02d}.csv'
    pd.DataFrame(out).to_csv(name, header=False, index=False)
    print(name, out.shape, 'anomalies', kf + ks, f'({(kf+ks)/len(out):.3%})  flip={kf} shuffle={ks}')
