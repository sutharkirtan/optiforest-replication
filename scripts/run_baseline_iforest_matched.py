"""scikit-learn IsolationForest on the 24 constructed files with per-seed results, under default sub-sampling (256 rows/tree)
and with matched sub-sampling (all rows per tree, as OptIForest uses).
Usage (repo root): python3 scripts/run_baseline_iforest_matched.py [seeds=10] -> results_v2_iforest_matched.json"""
import sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 10; res = {}
for b in ('ad', 'cc'):
    for t in ('global', 'local', 'cluster', 'shuffle'):
        for s in (1, 2, 3):
            n = f'{b}_{t}_s{s}'; d = pd.read_csv(f'data/v2/{n}.csv', header=None).values; X, y = d[:, :-1], d[:, -1]; res[n] = {}
            for name, ms in (('default', 256), ('full', len(X))):
                A, P = [], []
                for r in range(reps):
                    sc = -IsolationForest(n_estimators=100, max_samples=ms, random_state=r).fit(X).score_samples(X)
                    A.append(float(roc_auc_score(y, sc))); P.append(float(average_precision_score(y, sc)))
                res[n][name] = dict(auc=A, pr=P)
            print(n, {k: round(float(np.mean(v['auc'])), 3) for k, v in res[n].items()}, flush=True)
json.dump(res, open('results_v2_iforest_matched.json', 'w'))
