# OptIForest replication and extension

Replication of *OptIForest: Optimal Isolation Forest for Anomaly Detection* (Xiang et al., IJCAI 2023), plus new-data experiments.
Setup, commands, file layout and code changes: see `REPLICATION.md`. Tests: `cd tests && PYTHONPATH=.. python -m pytest -q .` (13 of 14 pass; the failure is a dimension defect in the test file).

## Results summary
| Item | Paper | Ours |
|---|---|---|
| AD, AUC-ROC (15 repeats) | 77.4 +/- 0.7 | 77.1 +/- 0.5 |
| AD, AUC-PR (15 repeats) | 43.8 +/- 3.1 | 42.8 +/- 3.7 |
| Credit Card cc_a (9.84% fraud), AUC-ROC / AUC-PR | - | 93.5 / 69.7 |
| Credit Card cc_b (2.0% fraud) | - | 95.2 / 55.3 |
| Credit Card 30k rows (0.17% fraud, 5 results from 3 executions) | - | 97.1 / 22.1 |
| 24 constructed files, mean AUC-ROC, OptIForest vs scikit-learn iForest (all rows per tree) | - | 0.666 vs 0.772 |
| Mixed (e) vs binary splits | e claimed optimal | within 0.01 on 21 of 24 files |

Environment used for the new runs: Python 3.13, NumPy 2.5, scikit-learn 1.9, 2 cores (see `requirements.txt`).

Credit Card files and anything derived from them are not in this repository (redistribution terms unconfirmed); scripts regenerate them from your own download. Same-seed OptIForest runs are not exactly repeatable (observed on the real-rate sample), so expect small differences.
