"""Subsample Credit Card Fraud (ULB) for OptIForest. Usage: python3 scripts/prep_creditcard.py path/to/creditcard.csv
cc_a: all 492 frauds + 4,508 random normals (5,000 rows, 9.8% fraud)
cc_b: 100 random frauds + 4,900 random normals (5,000 rows, 2.0% fraud)
Features (Time, V1-V28, Amount) standardised; headerless CSV, label last. Seed 42."""
import sys, os, numpy as np, pandas as pd; os.makedirs("data/existing", exist_ok=True)
d = pd.read_csv(sys.argv[1]); rng = np.random.default_rng(42)
fr = d[d.Class == 1]; nm = d[d.Class == 0]
def make(nf, nn, name):
    s = pd.concat([fr.sample(nf, random_state=42), nm.sample(nn, random_state=42)]).sample(frac=1, random_state=42)
    X = s.drop(columns='Class'); X = (X - X.mean()) / X.std()
    out = np.column_stack([X.values, s.Class.values]); pd.DataFrame(out).to_csv(f'data/existing/{name}.csv', header=False, index=False)
    print(name, out.shape, 'fraud', int(s.Class.sum()), f'{s.Class.mean():.3%}')
make(492, 4508, 'cc_a'); make(100, 4900, 'cc_b')
