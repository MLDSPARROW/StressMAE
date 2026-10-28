#!/usr/bin/env python3
"""Strict (LOSO) comparison in which every model gets exactly the same pipeline as StressMAE:
same 12 training / 2 validation subjects, post-processing and consensus-vote settings chosen on the 2 validation
subjects only, consensus clustering in the model's own representation (no labels of the test subject).
Output: results/final_strict.json (per-subject scores) and results/final_strict_stats.json (Wilcoxon + Holm)."""
import json, os
import numpy as np, torch
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import run_experiments as rx
import run_multi as rm
from run_experiments import SUBJECTS, WIN, SEED, batches
import select_eval as se

sn = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-6)
H = {s: v for s, v in np.load(os.path.join(rx.CACHE_DIR, "hrv_U.npz")).items()}
M = {s: rm.load_multi(s) for s in SUBJECTS}
FEAT = {s: sn(np.hstack([sn(H[s]), rm.feats(M[s]["U"])])) for s in SUBJECTS}      # handcrafted multi-sensor features


def split(i):
    others = [s for s in SUBJECTS if s != SUBJECTS[i]]
    val = [others[(i + j) % len(others)] for j in (0, 7)]
    return [s for s in others if s not in val], val


def feature_probs(make):
    """train on the 12 training subjects only, predict the test and validation recordings"""
    out = {}
    for i, test in enumerate(SUBJECTS):
        train, val = split(i)
        clf = make().fit(np.concatenate([FEAT[s][M[s]["t"] // WIN] for s in train]), np.concatenate([M[s]["y"] for s in train]))
        out[test] = (clf.predict_proba(FEAT[test]), {v: clf.predict_proba(FEAT[v]) for v in val})
    return out


_ECG = {}


def ecg_embeddings(test, subs):
    """frozen embeddings of the single-lead ECG StressMAE encoder of the test subject's group"""
    mae = rx.load_encoder(test); mae.eval(); out = {}
    for s in subs:
        if (test, s) not in _ECG:
            U = rx.load_subject(s)["U"]
            with torch.no_grad():
                E = torch.cat([mae.encode(torch.from_numpy(U[bi][:, None, :])) for bi in batches(len(U), 256, False)]).numpy()
            _ECG[(test, s)] = sn(E)
        out[s] = _ECG[(test, s)]
    return out


def evaluate(name, get, emb):
    """val-chosen post-processing (no clustering) and val-chosen consensus-vote for one model"""
    plain, cons = {}, {}
    for s in SUBJECTS:
        r = get(s)
        if r is None:
            continue
        Pt, Pv = r
        best = max(se.SETTINGS, key=lambda k: np.mean([se.score(v, se.SETTINGS[k](P))["f1"] for v, P in Pv.items()]))
        plain[s] = se.score(s, se.SETTINGS[best](Pt))
        E = emb(s, [s] + list(Pv))
        bc = max(se.CONS_SETTINGS, key=lambda c: np.mean([se.score(v, se.cons_apply(P, E[v], c, (name, s, v)))["f1"] for v, P in Pv.items()]))
        cons[s] = se.score(s, se.cons_apply(Pt, E[s], bc, (name, s, s)))
        cons[s]["setting"] = list(map(str, bc))
        raw = se.score(s, Pt); plain[s]["raw_f1"] = raw["f1"]; plain[s]["raw_acc"] = raw["acc"]
        print(f"  {name:28s} {s:4s} raw {raw['f1']:.3f} | post {plain[s]['f1']:.3f} | consensus {cons[s]['f1']:.3f}", flush=True)
    return plain, cons


def main():
    fprobs = {"Features + LR": feature_probs(lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))),
              "Features + RF": feature_probs(lambda: RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", n_jobs=4, random_state=SEED))}
    femb = lambda test, subs: {s: FEAT[s] for s in subs}
    models = [("StressMAE-multi", se.multi("StressMAE-multi"), lambda t, subs: se.multi_embeddings(t, subs)),
              ("No pre-training (multi)", se.multi("No-pretrain-multi"), femb),
              ("StressMAE ECG-only", se.ecg_ens, ecg_embeddings),
              ("Features + LR", lambda s: fprobs["Features + LR"][s], femb),
              ("Features + RF", lambda s: fprobs["Features + RF"][s], femb)]
    R = {}
    for name, get, emb in models:
        plain, cons = evaluate(name, get, emb)
        R[name] = {"post": plain, "consensus": cons}
    json.dump(R, open("results/final_strict.json", "w"), indent=1)
    # summary + paired tests of StressMAE-multi (consensus) against every other model (consensus), Holm-corrected
    ref = R["StressMAE-multi"]["consensus"]; stats = {}
    print("\nmean over 15 subjects (3-class acc/F1 | 2-class acc/F1)")
    for name in R:
        for v in ("post", "consensus"):
            S = R[name][v]; m = lambda k: np.mean([S[s][k] for s in S])
            print(f"  {name:28s} {v:10s} {m('acc'):.3f}/{m('f1'):.3f} | {m('bacc'):.3f}/{m('bf1'):.3f}"
                  + (f"   raw {m('raw_acc'):.3f}/{m('raw_f1'):.3f}" if v == "post" else ""))
    others = [n for n in R if n != "StressMAE-multi"]
    ps = []
    for n in others:
        x = np.array([ref[s]["f1"] - R[n]["consensus"][s]["f1"] for s in SUBJECTS])
        ps.append(wilcoxon(x).pvalue if np.any(x) else 1.0); stats[n] = {"mean_dF1": float(x.mean()), "wins": int((x > 0).sum()), "p": ps[-1]}
    order = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for r, j in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[j])); adj[j] = run
    for n, a in zip(others, adj):
        stats[n]["p_holm"] = float(a)
        print(f"  StressMAE-multi vs {n:26s} dF1 {stats[n]['mean_dF1']:+.3f} wins {stats[n]['wins']}/15 p={stats[n]['p']:.3f} Holm p={a:.3f}")
    # also: pre-training effect without clustering (post-processing only)
    x = np.array([R["StressMAE-multi"]["post"][s]["f1"] - R["No pre-training (multi)"]["post"][s]["f1"] for s in SUBJECTS])
    stats["pretraining_effect_post"] = {"mean_dF1": float(x.mean()), "wins": int((x > 0).sum()), "p": float(wilcoxon(x).pvalue)}
    print(f"  pre-training effect (post-processing only): dF1 {x.mean():+.3f}, wins {int((x > 0).sum())}/15, p={wilcoxon(x).pvalue:.3f}")
    json.dump(stats, open("results/final_strict_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
