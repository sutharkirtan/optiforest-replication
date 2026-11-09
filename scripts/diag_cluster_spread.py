"""Diagnostic: Credit Card 'cluster' anomalies at severity 3 (offset 4 sigma on 10 features) with varying blob spread.
Usage: PYTHONPATH=. python3 scripts/diag_cluster_spread.py path/to/creditcard.csv  -> results_diag_cluster.json"""
import sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from detectors import OptIForest
SEED = 42
d = pd.read_csv(sys.argv[1]); nm = d[d.Class == 0].drop(columns='Class')
B = nm.sample(2820, random_state=SEED).values; B = (B - B.mean(0)) / B.std(0)
n, m = B.shape; k = int(round(n * 0.05 / 0.95)); res = {}
for spread in (0.1, 0.5, 1.0):
    rng = np.random.default_rng(SEED + 1000 * 3 + 100 * 2 + 7)
    c = B[rng.integers(0, n)].copy(); j = rng.choice(m, 10, replace=False); c[j] += rng.choice([-1, 1], 10) * 4.0
    an = c + rng.normal(0, spread, (k, m)); X = np.vstack([B, an]); y = np.r_[np.zeros(n), np.ones(k)]
    A, P = [], []
    for _ in range(2):
        mdl = OptIForest('L2OPT', 100, 403, 0); mdl.fit(X); sc = -1.0 * mdl.decision_function(X)
        A.append(roc_auc_score(y, sc)); P.append(average_precision_score(y, sc))
    res[str(spread)] = dict(auc=float(np.mean(A)), pr=float(np.mean(P)), n_anom=int(k)); print(spread, res[str(spread)], flush=True)
json.dump(res, open('results_diag_cluster.json', 'w'), indent=1)
