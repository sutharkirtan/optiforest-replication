"""Run OptIForest (L2OPT, 100 trees, threshold 403, branch 0) on each constructed dataset; report AUC-ROC / AUC-PR per rate."""
import sys, time, warnings, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from detectors import OptIForest
runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
res = {}
for rate in ('01', '05', '10', '20'):
    d = pd.read_csv(f'data/constructed/ad_inj{rate}.csv', header=None).values
    X, y = d[:, :-1], d[:, -1]
    A, P, T = [], [], []
    for _ in range(runs):
        t = time.time()
        m = OptIForest('L2OPT', 100, 403, 0); m.fit(X)
        s = -1.0 * m.decision_function(X)
        T.append(time.time() - t)
        A.append(roc_auc_score(y, s)); P.append(average_precision_score(y, s))
    res[rate] = dict(auc=float(np.mean(A)), auc_std=float(np.std(A)), pr=float(np.mean(P)), pr_std=float(np.std(P)), base_rate=float(y.mean()), sec_per_run=float(np.mean(T)))
    print(rate, res[rate], flush=True)
json.dump(res, open('results_constructed.json', 'w'), indent=1)
