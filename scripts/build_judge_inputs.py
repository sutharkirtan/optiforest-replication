"""Build blinded inputs for the LLM-as-judge check on the Credit Card constructed files (cluster/global/local/shuffle, severities s1 and s3).
For each file: 16 evaluation rows (8 anomalies + 8 normals, shuffled) and 15 reference normal rows, rounded to 1 decimal.
Writes judge_input.json (no labels, file order shuffled) and judge_key.json (labels and file names) to the output directory.
Usage (repo root): python3 scripts/build_judge_inputs.py OUTDIR [seed=7]"""
import sys, json, numpy as np, pandas as pd
out = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7; rng = np.random.default_rng(seed)
names = [f'cc_{t}_s{s}' for t in ('global', 'local', 'cluster', 'shuffle') for s in (1, 3)]; rng.shuffle(names)
inp, key = [], []
for k, n in enumerate(names, 1):
    d = pd.read_csv(f'data/v2/{n}.csv', header=None).values; X, y = d[:, :-1], d[:, -1]
    an = rng.choice(np.where(y == 1)[0], 8, replace=False); nm = rng.choice(np.where(y == 0)[0], 23, replace=False)
    ev = np.concatenate([an, nm[:8]]); rng.shuffle(ev); ref = nm[8:]
    inp.append(dict(dataset=k, reference_normal_rows=np.round(X[ref], 1).tolist(),
                    rows_to_rate=[dict(id=f'{k}-{i+1}', features=np.round(X[r], 1).tolist()) for i, r in enumerate(ev)]))
    key.append(dict(dataset=k, file=n, labels={f'{k}-{i+1}': int(y[r]) for i, r in enumerate(ev)}))
json.dump(inp, open(f'{out}/judge_input.json', 'w')); json.dump(key, open(f'{out}/judge_key.json', 'w')); print(names)
