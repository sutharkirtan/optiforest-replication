# Constructed dataset: AD with injected anomalies
Base: normal class of UCI Internet Advertisements (Kushmerick, 1999): 2,820 rows x 1,555 binary features.
Format: headerless CSV, last column = label (0 normal, 1 injected anomaly), same as data/ad.csv.
Anomaly types (50/50): flip (normal row, 2% of bits flipped); shuffle (each feature drawn from a different random normal row).
Seed 42. Regenerate with: python3 scripts/build_constructed.py  (run from repo root)

| File | Rows | Anomalies | Rate |
|---|---|---|---|
| ad_inj01.csv | 2,848 | 28 | 0.98% |
| ad_inj05.csv | 2,968 | 148 | 4.99% |
| ad_inj10.csv | 3,133 | 313 | 9.99% |
| ad_inj20.csv | 3,525 | 705 | 20.0% |
