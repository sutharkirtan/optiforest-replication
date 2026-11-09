"""Focused e-vs-binary test: more repeats on the four constructed files with the largest mixed/binary gap.
Usage (repo root): PYTHONPATH=. python3 scripts/run_ebin_focus.py [repeats] -> results_ebin_focus.json"""
import sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from detectors import OptIForest
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 6; res = {}
for n in ('ad_cluster_s3', 'ad_shuffle_s1', 'ad_shuffle_s2', 'ad_shuffle_s3'):
    d = pd.read_csv(f'data/v2/{n}.csv', header=None).values; X, y = d[:, :-1], d[:, -1]; res[n] = {}
    for br in (0, 2):
        A = []
        for _ in range(reps):
            m = OptIForest('L2OPT', 100, 403, br); m.fit(X); A.append(roc_auc_score(y, -1.0 * m.decision_function(X)))
        res[n][f'branch{br}'] = dict(auc=float(np.mean(A)), sd=float(np.std(A)), runs=[float(a) for a in A])
    print(n, {k: (round(v['auc'], 4), round(v['sd'], 4)) for k, v in res[n].items()}, flush=True)
    json.dump(res, open('results_ebin_focus.json', 'w'), indent=1)
