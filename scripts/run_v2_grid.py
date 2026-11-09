"""All constructed files, both branch settings, per-run values saved (seeded: np.random.seed(1000*branch+repeat)).
Usage (repo root): PYTHONPATH=. python3 scripts/run_v2_grid.py <ad|cc> [repeats=5] -> results_v2_grid_<base>.json"""
import sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from detectors import OptIForest
b = sys.argv[1]; reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5; res = {}
for t in ('global', 'local', 'cluster', 'shuffle'):
    for s in (1, 2, 3):
        n = f'{b}_{t}_s{s}'; d = pd.read_csv(f'data/v2/{n}.csv', header=None).values; X, y = d[:, :-1], d[:, -1]; res[n] = {}
        for br in (0, 2):
            A, P = [], []
            for r in range(reps):
                np.random.seed(1000 * br + r); m = OptIForest('L2OPT', 100, 403, br); m.fit(X); sc = -1.0 * m.decision_function(X)
                A.append(float(roc_auc_score(y, sc))); P.append(float(average_precision_score(y, sc)))
            res[n][f'branch{br}'] = dict(auc=A, pr=P)
        print(n, {k: round(float(np.mean(v['auc'])), 4) for k, v in res[n].items()}, flush=True)
        json.dump(res, open(f'results_v2_grid_{b}.json', 'w'))
