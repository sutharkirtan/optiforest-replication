"""Credit Card at the real fraud rate (same 30,000-row sample as run_cc_real_rate.py, seed 42), with threshold-based metrics:
precision in the top 52 and top 100 alerts, and the best precision at recall >= 0.5.
Usage (repo root): PYTHONPATH=. python3 scripts/run_cc_real_rate_pk.py path/to/creditcard.csv [repeats=2] -> results_cc_real_rate_pk.json"""
import sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from detectors import OptIForest
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2; SEED = 42
d = pd.read_csv(sys.argv[1]); fr = d[d.Class == 1].sample(52, random_state=SEED); nm = d[d.Class == 0].sample(29948, random_state=SEED)
D = pd.concat([nm, fr]).sample(frac=1, random_state=SEED); y = D.Class.values; X = D.drop(columns='Class').values; X = (X - X.mean(0)) / X.std(0)
def metrics(sc):
    o = np.argsort(-sc); p, r, _ = precision_recall_curve(y, sc)
    return dict(auc=float(roc_auc_score(y, sc)), ap=float(average_precision_score(y, sc)), p_top52=float(y[o[:52]].mean()), p_top100=float(y[o[:100]].mean()), p_at_recall50=float(p[r >= 0.5].max()))
out = dict(opt=[], iforest=[])
for r in range(reps):
    np.random.seed(r); m = OptIForest('L2OPT', 100, 403, 0); m.fit(X); out['opt'].append(metrics(-1.0 * m.decision_function(X))); print('opt', r, out['opt'][-1], flush=True)
for r in range(5):
    out['iforest'].append(metrics(-IsolationForest(n_estimators=100, random_state=r).fit(X).score_samples(X))); print('if', r, out['iforest'][-1], flush=True)
json.dump(out, open('results_cc_real_rate_pk.json', 'w'), indent=1); print('done')
