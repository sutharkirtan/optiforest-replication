"""Score constructed 10% file separately for flip vs shuffle anomalies (rows after the 2,820 normals: first half flip, second half shuffle)."""
import sys, warnings, json; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from detectors import OptIForest
d = pd.read_csv('data/constructed/ad_inj10.csv', header=None).values; X, y = d[:, :-1], d[:, -1]
n = int((y == 0).sum()); k = len(y) - n; kf = k // 2
typ = np.array(['normal'] * n + ['flip'] * kf + ['shuffle'] * (k - kf))
res = {t: {'auc': [], 'pr': []} for t in ('flip', 'shuffle', 'all')}
for _ in range(3):
    m = OptIForest('L2OPT', 100, 403, 0); m.fit(X); s = -1.0 * m.decision_function(X)
    for t in ('flip', 'shuffle'):
        idx = (typ == 'normal') | (typ == t)
        res[t]['auc'].append(roc_auc_score(y[idx], s[idx])); res[t]['pr'].append(average_precision_score(y[idx], s[idx]))
    res['all']['auc'].append(roc_auc_score(y, s)); res['all']['pr'].append(average_precision_score(y, s))
out = {t: {k2: [float(np.mean(v)), float(np.std(v))] for k2, v in r.items()} for t, r in res.items()}
print(json.dumps(out, indent=1)); json.dump(out, open('results_by_type.json', 'w'), indent=1)
