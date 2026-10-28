#!/usr/bin/env python3
"""Second fine-tuning seed for the 'same network without pre-training' ablation, so that it is averaged over the
same number of seeds (2) as StressMAE. Same split, recipe and outputs as run_multi.py loso.
Output: results/multi_loso_extra/{S}.npz with keys No-pretrain-multi_s1 and No-pretrain-multi_s1|val_<S>."""
import argparse, os
import numpy as np, torch
import run_experiments as rx
import run_multi as rm
from run_experiments import SUBJECTS, SEED, set_seed

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--subjects", nargs="*", default=None); ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    out = os.path.join(rx.RES_DIR, "multi_loso_extra"); os.makedirs(out, exist_ok=True)
    data = {s: rm.load_multi(s) for s in SUBJECTS}
    for i, test in enumerate(SUBJECTS):
        if (a.subjects and test not in a.subjects) or os.path.exists(os.path.join(out, f"{test}.npz")):
            continue
        others = [s for s in SUBJECTS if s != test]
        val = [others[(i + j) % len(others)] for j in (0, 7)]; train = [s for s in others if s not in val]
        Xtr = np.concatenate([data[s]["X"] for s in train]); ytr = np.concatenate([data[s]["y"] for s in train])
        Xva = np.concatenate([data[s]["X"] for s in val]); yva = np.concatenate([data[s]["y"] for s in val])
        set_seed(SEED + 1000 * 1)
        model, ep = rm.fit(rm.Head(rm.MultiMAE()), Xtr, ytr, Xva, yva, 20, 3e-4)
        res = {"No-pretrain-multi_s1": rm.proba(model, data[test]["U"])}
        for v in val:
            res[f"No-pretrain-multi_s1|val_{v}"] = rm.proba(model, data[v]["U"])
        np.savez_compressed(os.path.join(out, f"{test}.npz"), **res)
        print(test, "done", ep, "epochs", flush=True)
