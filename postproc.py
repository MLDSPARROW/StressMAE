#!/usr/bin/env python3
"""Label-free post-processing of saved per-window probabilities (results/<dir>/S*_probs.npz), applied identically
to every method. Uses only the test person's own unlabelled recording, never its labels.
  raw    : argmax per 5 s window
  avg60  : causal mean of the last 12 windows
  zlogit : per-person logit standardisation, z = (log p - mean_rec) / std_rec   (removes the person's score offset)
  em     : label-shift EM (Saerens et al. 2002) for the person's class priors
  hmm    : Viterbi decoding with a sticky transition matrix, self-transition 1 - 1/D (D windows mean dwell time)
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import wilcoxon
import run_experiments as rx
from run_experiments import SUBJECTS, WIN
from sklearn.metrics import f1_score, accuracy_score

EPS = 1e-6


def avg(P, k=12):
    c = np.vstack([np.zeros((1, P.shape[1])), np.cumsum(P, 0)])
    i = np.arange(len(P)); lo = np.maximum(0, i - k + 1)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)[:, None]


def zlogit(P):
    L = np.log(P + EPS)
    return (L - L.mean(0)) / (L.std(0) + EPS)


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)


def em(P, iters=50):
    prior = np.full(P.shape[1], 1 / P.shape[1]); Q = P
    for _ in range(iters):
        Q = P * prior * P.shape[1]; Q /= Q.sum(1, keepdims=True)
        prior = Q.mean(0)
    return Q


def viterbi(P, D=60):
    n, k = P.shape
    a = 1 - 1 / D
    T = np.log(np.full((k, k), (1 - a) / (k - 1)) + np.eye(k) * (a - (1 - a) / (k - 1)))
    E = np.log(P + EPS)
    d = E[0].copy(); bp = np.zeros((n, k), int)
    for t in range(1, n):
        s = d[:, None] + T
        bp[t] = s.argmax(0); d = s.max(0) + E[t]
    path = np.zeros(n, int); path[-1] = d.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return np.eye(k)[path]


def is_scores(P):
    return P.min() < 0 or np.abs(P.sum(1) - 1).max() > 1e-3


def to_prob(P):
    return softmax(P) if is_scores(P) else np.clip(P, 0, 1)


TRANSFORMS = {
    "raw": lambda P: P,
    "avg60": lambda P: avg(P),
    "zlogit": lambda P: zlogit(P),
    "zlogit+avg60": lambda P: avg(softmax(zlogit(P))),
    "em": lambda P: em(P),
    "em+avg60": lambda P: avg(em(P)),
    "hmm": lambda P: viterbi(P),
    "zlogit+hmm": lambda P: viterbi(softmax(zlogit(P))),
}


def main(d="loso2", methods=None):
    data = {s: rx.load_subject(s) for s in SUBJECTS}
    files = sorted(glob.glob(f"results/{d}/S*_probs.npz"))
    R = {}
    for f in files:
        s = os.path.basename(f).split("_")[0]
        z = np.load(f); idx = data[s]["t"] // WIN; y = data[s]["y"]
        for m in z.files:
            if methods and m not in methods:
                continue
            P = to_prob(z[m])
            for tn, tf in TRANSFORMS.items():
                pr = tf(P)[idx].argmax(1)
                R.setdefault(m, {}).setdefault(tn, {})[s] = (f1_score(y, pr, average="macro"), accuracy_score(y, pr))
    subs = sorted({s for m in R for t in R[m] for s in R[m][t]})
    print(d, len(subs), "subjects; macro F1 (accuracy)")
    print(f"{'method':42s}" + "".join(f"{t:>14s}" for t in TRANSFORMS))
    for m in sorted(R, key=lambda m: -np.mean([v[0] for v in R[m]['raw'].values()])):
        print(f"{m[:42]:42s}" + "".join(f"{np.mean([v[0] for v in R[m][t].values()]):7.3f}({np.mean([v[1] for v in R[m][t].values()]):.2f})"
                                          for t in TRANSFORMS))
    json.dump({m: {t: {s: v for s, v in R[m][t].items()} for t in R[m]} for m in R},
              open(f"results/postproc_{d}.json", "w"))
    return R


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "loso2")
