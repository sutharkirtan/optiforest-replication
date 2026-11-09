"""Run OptIForest (L2OPT, 100 trees, threshold 403) on constructed v2 files with branch=0 (mixed, avg e) and branch=2 (binary).
Usage: PYTHONPATH=. python3 scripts/run_v2.py <ad|cc> [repeats]   -> results_v2_<base>.json"""
import sys, time, warnings, json; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from detectors import OptIForest
base = sys.argv[1]; reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2; res = {}
for t in ('global', 'local', 'cluster', 'shuffle'):
    for s in (1, 2, 3):
        name = f'{base}_{t}_s{s}'; d = pd.read_csv(f'data/v2/{name}.csv', header=None).values; X, y = d[:, :-1], d[:, -1]
        res[name] = {}
        for br in (0, 2):
            A, P = [], []
            for _ in range(reps):
                m = OptIForest('L2OPT', 100, 403, br); m.fit(X); sc = -1.0 * m.decision_function(X)
                A.append(roc_auc_score(y, sc)); P.append(average_precision_score(y, sc))
            res[name][f'branch{br}'] = dict(auc=float(np.mean(A)), pr=float(np.mean(P)), auc_sd=float(np.std(A)), pr_sd=float(np.std(P)))
        print(name, res[name], flush=True)
        json.dump(res, open(f'results_v2_{base}.json', 'w'), indent=1)
