"""Credit Card cluster anomalies with severity controlled by blob spread (offset crossed with spread), OptIForest (mixed) and iForest.
Usage (repo root): PYTHONPATH=. python3 scripts/run_cluster_spread.py path/to/creditcard.csv [repeats=5] -> results_cluster_spread.json"""
import sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from detectors import OptIForest
SEED = 42; reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
d = pd.read_csv(sys.argv[1]); nm = d[d.Class == 0].drop(columns='Class')
B = nm.sample(2820, random_state=SEED).values; B = (B - B.mean(0)) / B.std(0); n, m = B.shape; k = int(round(n * 0.05 / 0.95)); res = {}
for off in (1.5, 4.0):
    for spread in (0.1, 0.25, 0.5, 0.75, 1.0, 1.5):
        rng = np.random.default_rng(SEED + 1000 * 3 + 100 * 2 + 7)
        c = B[rng.integers(0, n)].copy(); j = rng.choice(m, 10, replace=False); c[j] += rng.choice([-1, 1], 10) * off
        X = np.vstack([B, c + rng.normal(0, spread, (k, m))]); y = np.r_[np.zeros(n), np.ones(k)]
        oa, op, fa, fp = [], [], [], []
        for r in range(reps):
            np.random.seed(r); mdl = OptIForest('L2OPT', 100, 403, 0); mdl.fit(X); sc = -1.0 * mdl.decision_function(X)
            oa.append(float(roc_auc_score(y, sc))); op.append(float(average_precision_score(y, sc)))
            sc = -IsolationForest(n_estimators=100, random_state=r).fit(X).score_samples(X)
            fa.append(float(roc_auc_score(y, sc))); fp.append(float(average_precision_score(y, sc)))
        key = f'off{off}_spread{spread}'; res[key] = dict(opt_auc=oa, opt_pr=op, if_auc=fa, if_pr=fp)
        print(key, round(float(np.mean(oa)), 3), round(float(np.mean(fa)), 3), flush=True)
        json.dump(res, open('results_cluster_spread.json', 'w'))
