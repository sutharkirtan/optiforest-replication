"""Repeatability check: fit OptIForest on the first 1,500 rows of data/ad.csv with np.random.seed(0); print AUC-ROC, AP and a hash of the scores.
Run it several times as separate processes (repo root: PYTHONPATH=. python3 scripts/check_determinism.py); identical output means repeatable.
In our environment AUC-ROC (0.794-0.802), AP (0.511-0.550) and the hash differed on every invocation, with and without PYTHONHASHSEED set."""
import numpy as np, pandas as pd, hashlib, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score
from detectors import OptIForest
a = pd.read_csv('data/ad.csv', header=None).values[:1500]; X, y = a[:, :-1], a[:, -1]
np.random.seed(0); m = OptIForest('L2OPT', 100, 403, 0); m.fit(X); s = -1.0 * np.asarray(m.decision_function(X)).ravel()
print(round(roc_auc_score(y, s), 4), round(average_precision_score(y, s), 4), hashlib.md5(s.tobytes()).hexdigest())
