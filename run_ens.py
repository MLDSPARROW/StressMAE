#!/usr/bin/env python3
"""StressMAE (same encoder, same fine-tuning as round 1) with several fine-tuning seeds. Saves the per-window
probabilities of the whole test recording AND of the two validation subjects, so that the post-processing
setting (dwell time D) can be chosen on validation subjects only. Output: results/ens/{S}_seed{k}.npz"""
import argparse, copy, os, time
import numpy as np, torch
import run_experiments as rx
from run_experiments import SUBJECTS, SEED, MAEClassifier, set_seed, fit, predict_proba, load_encoder

OUT = os.path.join(rx.RES_DIR, "ens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--ft-epochs", type=int, default=20)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); os.makedirs(OUT, exist_ok=True)
    data = {s: rx.load_subject(s) for s in SUBJECTS}
    for i, test in enumerate(SUBJECTS):
        if a.subjects and test not in a.subjects:
            continue
        others = [s for s in SUBJECTS if s != test]
        val = [others[(i + j) % len(others)] for j in (0, 7)]          # same split as run_experiments.py
        train = [s for s in others if s not in val]
        Xtr = np.concatenate([data[s]["X"] for s in train]); ytr = np.concatenate([data[s]["y"] for s in train])
        Xva = np.concatenate([data[s]["X"] for s in val]); yva = np.concatenate([data[s]["y"] for s in val])
        mae = load_encoder(test)
        for k in a.seeds:
            path = os.path.join(OUT, f"{test}_seed{k}.npz")
            if os.path.exists(path):
                continue
            t0 = time.time(); set_seed(SEED + 1000 * k)
            model, ep = fit(MAEClassifier(copy.deepcopy(mae)), Xtr, ytr, Xva, yva, a.ft_epochs, 1e-4)
            out = {"test": predict_proba(model, data[test]["U"])}
            for v in val:
                out[f"val_{v}"] = predict_proba(model, data[v]["U"])
            np.savez_compressed(path, **out)
            print(f"{test} seed {k} done ({ep} ep, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
