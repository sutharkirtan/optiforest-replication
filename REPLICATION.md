# OptIForest replication and extension (COMP8240 final project)

This repository is the authors' OptIForest code (Xiang et al., IJCAI 2023) with the small changes listed below, plus the
scripts, data generators and results used in the report. Results summary and quick start: `README.md`.

## Setup
```
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=.                                       # Windows PowerShell: $env:PYTHONPATH="."   (needed in every new window)
```

## 1. Replication on AD (paper, Table 1)
```
python scripts/replicate_short.py --dataset=ad --threshold=403 --branch=0     # 3 repeats, a few minutes
python demo.py                                                                   # full 15-repeat demo, about 14 minutes
```
Report values (AUC-ROC / AUC-PR): full 15-repeat demo 0.7712 / 0.4283 (see results_demo_full.log); short runs 0.7731 / 0.4214 and 0.7690 / 0.3971; paper 0.774 / 0.438.

## 2. Existing data: ULB Credit Card Fraud (not in the paper)
Source: Kaggle "Credit Card Fraud Detection" (ULB / Worldline), `creditcard.csv`, 150,828,752 bytes, SHA-256 `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` (the copy used in the report; check yours matches). Credit Card files and anything derived from them (cc_a, cc_b, data/v2/cc_*.csv, the 30,000-row sample) are not included in this repository; every script regenerates them with seed 42.

Download `creditcard.csv` from Kaggle ("Credit Card Fraud Detection", ULB / Worldline; check the licence on the page). Then:
```
python scripts/prep_creditcard.py path/to/creditcard.csv          # writes data/existing/cc_a.csv, cc_b.csv (not committed: derived from the Kaggle data, redistribution terms unconfirmed)
python scripts/run_existing.py 3                                  # results_existing.json
```

## 3. Constructed benchmark v2 (24 files)
```
python scripts/build_constructed_v2.py path/to/creditcard.csv     # writes data/v2/*.csv (about 116 MB) + manifest.csv (seed 42)
python scripts/run_v2_grid.py ad 5   &&   python scripts/run_v2_grid.py cc 3   # OptIForest, branch 0 (mixed) and 2 (binary); AD about 100 min, CC about 60 min on 1 core
python scripts/summarise_grid.py results_v2_grid_ad.json results_v2_grid_cc.json   # means, sd, 95% CI of mixed minus binary
python scripts/run_baseline_iforest.py 10                                   # scikit-learn IsolationForest baseline
python scripts/run_heldout.py 3                                             # 50/50 held-out check on ad, cc_a, cc_b
python scripts/run_cluster_spread.py path/to/creditcard.csv 3               # offset x spread grid, OptIForest and iForest
python scripts/run_cc_real_rate.py path/to/creditcard.csv 2                 # 30,000 rows (52 frauds + 29,948 normals, seed 42) at the real 0.17% rate
python scripts/run_cc_real_rate_pk.py path/to/creditcard.csv 2              # same sample (seeds 0,1 again): precision at top-k and at 50% recall.
# The report's five real-rate OptIForest values: two executions of run_cc_real_rate.py (the second stopped after seed 0; see results_cc_real_rate_third.json) and one of run_cc_real_rate_pk.py. Same seeds, different AP. Also: python scripts/check_determinism.py (compare hashes across invocations)
python scripts/run_baseline_iforest_matched.py 10                           # isolation forest with default (256) and matched (all rows) sub-sampling
python scripts/diag_cluster_spread.py path/to/creditcard.csv                # earlier single-offset diagnostic
```
Files are named `<base>_<type>_s<severity>.csv`; headerless, label in the last column. `data/v2/manifest.csv` is the annotation
(file, row, is_anomaly, type, severity). `data/v2/README.md` is the data card. Regenerated files are byte-identical to those used in the report.
Result files: `results_v2_grid_ad.json`, `results_v2_grid_cc.json` (per-run values used in the report), `results_v2_ad.json`, `results_v2_cc.json` (earlier 2-repeat runs), `results_v2_iforest.json`, `results_heldout.json`, `results_cluster_spread.json`, `results_cc_real_rate.json`, `results_cc_real_rate_pk.json`, `results_v2_iforest_matched.json`, `results_diag_cluster.json`, `results_existing.json`.

## Changes to the authors' code
- `detectors/opt_tree.py` (predict): `np.mat(point)` replaced by `np.asarray(point)` (np.mat was removed in NumPy 2).
- `detectors/opt_forest.py` (bagging path): `np.mat` replaced by `np.asmatrix`.
## Tests
```
cd tests && PYTHONPATH=.. python -m pytest -q .
```
13 of 14 pass. `TestHierHash::test_get_hash_value` fails because the test builds a 1556-dimensional input for a 1555-dimensional model (a defect in the test).

## LLM-as-judge check
```
python scripts/build_judge_inputs.py OUTDIR 7     # blinded input + key (needs data/v2/cc_*.csv)
# give OUTDIR/judge_input.json to a fresh LLM judge (prompt in the report appendix); save its output as OUTDIR/judge_output.json
python scripts/score_judge.py OUTDIR              # writes results_llm_judge.json
```
Our run: `judge/judge_key.json`, `judge/judge_output.json`, `results_llm_judge.json`.
