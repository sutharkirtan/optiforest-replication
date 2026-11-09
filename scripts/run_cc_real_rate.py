"""Credit Card at the real fraud rate: 30,000 rows with 52 frauds (0.173%), standardised; OptIForest (mixed) x repeats and iForest.
Usage (repo root): PYTHONPATH=. python3 scripts/run_cc_real_rate.py path/to/creditcard.csv [repeats=3] -> results_cc_real_rate.json"""
import sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from detectors import OptIForest
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3; SEED = 42
d = pd.read_csv(sys.argv[1]); fr = d[d.Class == 1].sample(52, random_state=SEED); nm = d[d.Class == 0].sample(29948, random_state=SEED)
D = pd.concat([nm, fr]).sample(frac=1, random_state=SEED); y = D.Class.values; X = D.drop(columns='Class').values; X = (X - X.mean(0)) / X.std(0)
oa, op, fa, fp = [], [], [], []
for r in range(reps):
    np.random.seed(r); m = OptIForest('L2OPT', 100, 403, 0); m.fit(X); sc = -1.0 * m.decision_function(X)
    oa.append(float(roc_auc_score(y, sc))); op.append(float(average_precision_score(y, sc))); print('opt', r, oa[-1], op[-1], flush=True)
for r in range(5):
    sc = -IsolationForest(n_estimators=100, random_state=r).fit(X).score_samples(X); fa.append(float(roc_auc_score(y, sc))); fp.append(float(average_precision_score(y, sc)))
json.dump(dict(n=int(len(y)), frauds=int(y.sum()), rate=float(y.mean()), opt_auc=oa, opt_pr=op, if_auc=fa, if_pr=fp), open('results_cc_real_rate.json', 'w'), indent=1); print('done')
