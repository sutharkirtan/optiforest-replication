"""Baseline: scikit-learn IsolationForest (100 trees, default max_samples=256) on the 24 constructed v2 files.
Usage (repo root): python3 scripts/run_baseline_iforest.py [repeats]  -> results_v2_iforest.json"""
import sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5; res = {}
for b in ('ad', 'cc'):
    for t in ('global', 'local', 'cluster', 'shuffle'):
        for s in (1, 2, 3):
            n = f'{b}_{t}_s{s}'; d = pd.read_csv(f'data/v2/{n}.csv', header=None).values; X, y = d[:, :-1], d[:, -1]; A, P = [], []
            for r in range(reps):
                sc = -IsolationForest(n_estimators=100, random_state=r).fit(X).score_samples(X)
                A.append(roc_auc_score(y, sc)); P.append(average_precision_score(y, sc))
            res[n] = dict(auc=float(np.mean(A)), pr=float(np.mean(P)), auc_sd=float(np.std(A)), pr_sd=float(np.std(P))); print(n, res[n], flush=True)
json.dump(res, open('results_v2_iforest.json', 'w'), indent=1)
