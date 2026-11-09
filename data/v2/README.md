# Constructed benchmark v2 (typed, graded anomaly injection)
Regenerate everything: `python3 scripts/build_constructed_v2.py path/to/creditcard.csv` (seed 42; run from repo root, needs data/ad.csv).
The CSVs are not committed (hundreds of MB); the generator and `manifest.csv` are.

## Bases (clean, normal class only)
- **ad**: UCI Internet Advertisements (Kushmerick, 1999), 2,820 rows x 1,555 binary features, ~0.7% density
- **cc**: ULB Credit Card Fraud (Kaggle), 2,820 random normal transactions x 30 features (Time, V1-V28, Amount), standardised

## Anomaly types (5% contamination = 148 anomalies per file), severity 1 (subtle) to 3 (strong)
| Type | AD (binary) | CC (continuous) | What it tests |
|---|---|---|---|
| global | flip 0.5 / 1 / 2% of bits | 5 features shifted by 2 / 3.5 / 5 sigma | classic outliers |
| local | swap a block of 5 / 15 / 30% of columns with another row's | 10 features moved by 0.8 / 1.5 / 2.5 sigma | anomalies inside the normal envelope |
| cluster | one prototype copied with 0.5 / 0.2 / 0.1% flips | tight Gaussian blob offset 1.5 / 2.5 / 4 sigma | collective anomalies (masking) |
| shuffle | 30 / 60 / 100% of features drawn from other rows | same | marginals kept, correlations destroyed |

## Annotation
Labels come from construction. `manifest.csv` records, for every row of every file: file, row, is_anomaly, type, severity.
Files: `<base>_<type>_s<severity>.csv`, headerless, label in last column (same format as data/ad.csv).
