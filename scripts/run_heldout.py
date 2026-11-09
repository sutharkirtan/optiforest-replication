"""Held-out evaluation: fit (unsupervised) on a random stratified 50% of rows, score the other 50%; compare with scoring the training half.
Usage (repo root): PYTHONPATH=. python3 scripts/run_heldout.py [repeats=3] -> results_heldout.json"""
import sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from detectors import OptIForest
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3; res = {}
for name, path in (('AD', 'data/ad.csv'), ('cc_a', 'data/existing/cc_a.csv'), ('cc_b', 'data/existing/cc_b.csv')):
    d = pd.read_csv(path, header=None).values; X, y = d[:, :-1], d[:, -1]; rows = []
    for r in range(reps):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=r)
        np.random.seed(r); m = OptIForest('L2OPT', 100, 403, 0); m.fit(Xtr)
        s_tr, s_te = -1.0 * m.decision_function(Xtr), -1.0 * m.decision_function(Xte)
        rows.append(dict(train_auc=float(roc_auc_score(ytr, s_tr)), train_pr=float(average_precision_score(ytr, s_tr)),
                         test_auc=float(roc_auc_score(yte, s_te)), test_pr=float(average_precision_score(yte, s_te))))
    res[name] = rows; print(name, {k: round(float(np.mean([x[k] for x in rows])), 4) for k in rows[0]}, flush=True)
    json.dump(res, open('results_heldout.json', 'w'), indent=1)
