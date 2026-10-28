#!/usr/bin/env python3
"""Non-strict (same person in training) baselines with exactly the splits used for StressMAE-multi in run_multi.py cv:
random stratified 10-fold inside each subject (3 and 4 classes) and the chronological 2/3-1/3 split per condition.
Handcrafted multi-sensor features (19 ECG/HRV + 25 channel statistics) with LR and RF. Output results/final_nonstrict_baselines.json"""
import json, os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import run_experiments as rx
import run_multi as rm
from run_experiments import SUBJECTS, WIN, SEED

sn = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-6)
H = {s: v for s, v in np.load(os.path.join(rx.CACHE_DIR, "hrv_U.npz")).items()}
MODELS = {"Features + LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")),
          "Features + RF": lambda: RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", n_jobs=4, random_state=SEED)}


def run(classes, chrono):
    out = {}
    for s in SUBJECTS:
        d = rm.load_multi(s, classes); y = d["y"]; X = sn(np.hstack([sn(H[s]), rm.feats(d["U"])]))[d["t"] // WIN]
        if chrono:
            tr_m = np.zeros(len(y), bool)
            for c in np.unique(y):
                ii = np.where(y == c)[0]; tr_m[ii[: int(round(len(ii) * 2 / 3))]] = True
            splits = [(np.where(tr_m)[0], np.where(~tr_m)[0])]
        else:
            splits = list(StratifiedKFold(10, shuffle=True, random_state=SEED).split(X, y))
        ev = np.concatenate([te for _, te in splits])
        for name, mk in MODELS.items():
            pred = np.zeros(len(y), int)
            for tr, te in splits:
                pred[te] = mk().fit(X[tr], y[tr]).predict(X[te])
            p, yy = pred[ev], y[ev]
            out.setdefault(name, {})[s] = {"acc": float((p == yy).mean()), "f1": f1_score(yy, p, average="macro")}
    return out


if __name__ == "__main__":
    R = {"cv3": run((1, 2, 3), False), "cv4": run((1, 2, 3, 4), False), "chrono3": run((1, 2, 3), True)}
    json.dump(R, open("results/final_nonstrict_baselines.json", "w"), indent=1)
    for k, v in R.items():
        for n, S in v.items():
            print(k, n, f"acc {np.mean([x['acc'] for x in S.values()]):.4f} F1 {np.mean([x['f1'] for x in S.values()]):.4f}")
