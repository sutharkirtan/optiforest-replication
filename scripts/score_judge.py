"""Score the blinded LLM judge: per-file AUC-ROC of the judge's 1-10 scores against the construction labels,
beside the detectors' AUC on the same files. Usage (repo root): python3 scripts/score_judge.py JUDGE_DIR -> results_llm_judge.json"""
import sys, json, numpy as np
from sklearn.metrics import roc_auc_score
d = sys.argv[1]; key = json.load(open(f'{d}/judge_key.json')); out = json.load(open(f'{d}/judge_output.json'))
cc = json.load(open('results_v2_grid_cc.json')); im = json.load(open('results_v2_iforest_matched.json')); res = {}
for k in key:
    ids = list(k['labels']); y = [k['labels'][i] for i in ids]; sc = [out[str(k['dataset'])][i]['score'] for i in ids]
    f = k['file']
    res[f] = dict(judge_auc=float(roc_auc_score(y, sc)), judge_mean_anom=float(np.mean([s for s, l in zip(sc, y) if l])), judge_mean_norm=float(np.mean([s for s, l in zip(sc, y) if not l])),
                  optiforest_auc=float(np.mean(cc[f]['branch0']['auc'])), iforest_all_auc=float(np.mean(im[f]['full']['auc'])), n=len(y), n_anom=int(sum(y)), judge_scores=dict(zip(ids, sc)))
    print(f"{f:14s} judge {res[f]['judge_auc']:.3f} (anom {res[f]['judge_mean_anom']:.1f} vs norm {res[f]['judge_mean_norm']:.1f})  OptIForest {res[f]['optiforest_auc']:.3f}  iForest(all rows) {res[f]['iforest_all_auc']:.3f}")
json.dump(res, open('results_llm_judge.json', 'w'), indent=1)
