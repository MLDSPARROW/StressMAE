#!/usr/bin/env python3
"""Honest strict (LOSO) evaluation with label-free post-processing.
Our models: the post-processing setting is chosen per test subject on its 2 VALIDATION subjects only.
Baselines without validation outputs get their best setting chosen on the TEST subjects (favours the baselines).
Usage: python select_eval.py"""
import glob, json, os
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, accuracy_score
import run_experiments as rx
from run_experiments import SUBJECTS, WIN
from postproc import zlogit, softmax, viterbi, avg, to_prob

SETTINGS = {"raw": lambda P: P, "avg60": lambda P: avg(P), "zlogit": lambda P: softmax(zlogit(P)),
            "zlogit+avg60": lambda P: avg(softmax(zlogit(P)))}
for D in (12, 24, 60, 120, 240):
    SETTINGS[f"hmm{D}"] = (lambda D: lambda P: viterbi(P, D))(D)
    SETTINGS[f"zlogit+hmm{D}"] = (lambda D: lambda P: viterbi(softmax(zlogit(P)), D))(D)

NCLS = int(os.environ.get("NCLS", 3))
if NCLS in (2, 3):
    DATA = {s: rx.load_subject(s) for s in SUBJECTS}
    if NCLS == 2:
        for s in SUBJECTS:
            DATA[s]["y"] = (DATA[s]["y"] == 1).astype(int)
else:   # 4 classes (baseline, stress, amusement, meditation): labels from the multi-channel cache
    import run_multi
    DATA = {s: {k: v for k, v in run_multi.load_multi(s, (1, 2, 3, 4)).items() if k in ("y", "t")} for s in SUBJECTS}


def score(s, P):
    d = DATA[s]; pr = P[d["t"] // WIN].argmax(1); y = d["y"]
    yb, pb = (y == 1).astype(int), (pr == 1).astype(int)
    return {"f1": f1_score(y, pr, average="macro"), "acc": accuracy_score(y, pr),
            "bf1": f1_score(yb, pb, average="macro"), "bacc": accuracy_score(yb, pb)}


def val_selected(get):
    """get(s) -> (P_test, {val_subject: P_val}). Returns per-subject scores with the val-chosen setting."""
    out, chosen = {}, {}
    for s in SUBJECTS:
        r = get(s)
        if r is None:
            continue
        Pt, Pv = r
        best = max(SETTINGS, key=lambda k: np.mean([score(v, SETTINGS[k](P))["f1"] for v, P in Pv.items()]))
        chosen[s] = best; out[s] = score(s, SETTINGS[best](Pt))
    return out, chosen


def oracle(get):
    """best single setting chosen on the test subjects themselves (optimistic for that method)"""
    per = {}
    for s in SUBJECTS:
        P = get(s)
        if P is not None:
            per[s] = {k: score(s, f(P)) for k, f in SETTINGS.items()}
    best = max(SETTINGS, key=lambda k: np.mean([per[s][k]["f1"] for s in per]))
    return {s: per[s][best] for s in per}, best


def ecg_ens(s):
    fs = sorted(glob.glob(f"results/ens/{s}_seed*.npz"))
    if not fs:
        return None
    Z = [np.load(f) for f in fs]
    Pt = np.mean([z["test"] for z in Z], 0)
    vals = [k[4:] for k in Z[0].files if k.startswith("val_")]
    return Pt, {v: np.mean([z[f"val_{v}"] for z in Z], 0) for v in vals}


def multi(prefix, d="multi_loso"):
    def get(s):
        f = f"results/{d}/{s}.npz"
        if not os.path.exists(f):
            return None
        z = dict(np.load(f))
        fx = f"results/multi_loso_extra/{s}.npz"     # extra fine-tuning seeds (same split), if present
        if d == "multi_loso" and os.path.exists(fx):
            z.update(dict(np.load(fx)))
        z = type("Z", (), {"files": list(z), "__getitem__": lambda self, k, z=z: z[k]})()
        keys = [k for k in z.files if k.startswith(prefix) and "|" not in k]
        if not keys:
            return None
        Pt = np.mean([z[k] for k in keys], 0)
        vals = sorted({k.split("|val_")[1] for k in z.files if k.startswith(prefix) and "|val_" in k})
        return Pt, {v: np.mean([z[f"{k}|val_{v}"] for k in keys], 0) for v in vals}
    return get


def two_view(s):
    """StressMAE two-view ensemble: ECG-only StressMAE + multi-channel StressMAE. The mixing weight w and the
    post-processing setting are chosen together on the 2 validation subjects."""
    a, b = ecg_ens(s), multi("StressMAE-multi")(s)
    if a is None or b is None:
        return None
    return a, b


def val_selected_two_view():
    out, chosen = {}, {}
    for s in SUBJECTS:
        r = two_view(s)
        if r is None:
            continue
        (Ea, Va), (Eb, Vb) = r
        cands = [(w, k) for w in (0.0, 0.25, 0.5, 0.75, 1.0) for k in SETTINGS]
        best = max(cands, key=lambda c: np.mean([score(v, SETTINGS[c[1]](c[0] * Vb[v] + (1 - c[0]) * Va[v]))["f1"] for v in Va]))
        chosen[s] = best; out[s] = score(s, SETTINGS[best[1]](best[0] * Eb + (1 - best[0]) * Ea))
    return out, chosen


def baseline(path_fmt, key):
    def get(s):
        f = path_fmt.format(s)
        if not os.path.exists(f):
            return None
        z = np.load(f)
        return to_prob(z[key]) if key in z.files else None
    return get


def summary(name, res, extra=""):
    S = sorted(res); m = lambda k: np.mean([res[s][k] for s in S])
    print(f"{name:44s} n={len(S):2d}  3-cl acc {m('acc'):.3f} F1 {m('f1'):.3f} | 2-cl acc {m('bacc'):.3f} F1 {m('bf1'):.3f} {extra}")


def main():
    rows = {}
    if NCLS != 3:
        for name, pre, d in [("StressMAE-multi (2 seeds)", "StressMAE-multi", f"multi_loso_plain{NCLS}"),
                             ("StressMAE-multi-cd (2 seeds)", "StressMAE-multi-cd", f"multi_loso_cd{NCLS}")]:
            res, ch = val_selected(multi(pre, d))
            if res:
                rows[name] = res; summary(f"{name} {NCLS}-class", res)
        json.dump(rows, open(f"results/select_eval{NCLS}.json", "w"), indent=1)
        return
    for name, get in [("StressMAE (ECG, 3 seeds)", ecg_ens), ("StressMAE-multi (2 seeds)", multi("StressMAE-multi")),
                      ("StressMAE-multi-cd (2 seeds)", multi("StressMAE-multi-cd", "multi_loso_cd")),
                      ("No-pretrain-multi", multi("No-pretrain-multi"))]:
        res, ch = val_selected(get)
        if res:
            rows[name] = res; summary(name, res, "[val-chosen: " + ", ".join(sorted(set(ch.values()))) + "]")
    res, ch = val_selected_two_view()
    if res:
        rows["StressMAE two-view (ECG + multi)"] = res
        summary("StressMAE two-view (ECG + multi)", res, "[val-chosen w: " + ", ".join(f"{s}:{c[0]}" for s, c in sorted(ch.items())) + "]")
    for name, get in [("Features-multi+LR (best on test)", baseline("results/multi_loso/{}.npz", "Features-multi+LR")),
                      ("Features-multi+RF (best on test)", baseline("results/multi_loso/{}.npz", "Features-multi+RF")),
                      ("HRV+LR+SN (best on test)", baseline("results/loso2/{}_probs.npz", "HRV+LR_+_SN")),
                      ("HRV+SVM+SN (best on test)", baseline("results/loso2/{}_probs.npz", "HRV+SVM_+_SN")),
                      ("No pre-training ECG (best on test)", baseline("results/loso2/{}_probs.npz", "No_pre-training")),
                      ("Supervised 1D-CNN (best on test)", baseline("results/loso2/{}_probs.npz", "Supervised_1D-CNN"))]:
        try:
            res, best = oracle(get)
        except (ValueError, ZeroDivisionError):
            continue
        rows[name] = res; summary(name, res, f"[{best}]")
    ours = [k for k in rows if k.startswith("StressMAE")]
    for o in ours:
        for b in rows:
            if b == o:
                continue
            S = sorted(set(rows[o]) & set(rows[b]))
            if len(S) >= 6:
                x = np.array([rows[o][s]["f1"] - rows[b][s]["f1"] for s in S])
                p = wilcoxon(x).pvalue if np.any(x) else 1.0
                print(f"  {o} vs {b}: mean dF1 {x.mean():+.3f}, wins {int((x > 0).sum())}/{len(S)}, p={p:.3f}")
    json.dump({k: v for k, v in rows.items()}, open("results/select_eval.json", "w"), indent=1)


if __name__ == "__main__" and not os.environ.get("CONSENSUS"):
    main()


# ---------------------------------------------------------------- cluster-then-vote (label-free, transductive)
def cluster_vote(P, E, k, seed=0):
    """k-means on the person's own frozen StressMAE embeddings (PCA 16); every cluster takes the class with the
    highest mean (calibrated) probability inside the cluster. Uses no labels of the person."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    Z = PCA(16, random_state=seed).fit_transform(E)
    lab = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
    Q = np.zeros_like(P)
    for c in range(k):
        m = lab == c
        Q[m] = P[m].mean(0)
    return Q


_EMB = {}


def multi_embeddings(test, subs):
    """frozen embeddings of the test subject's group encoder, per-subject standardised"""
    import torch, run_multi as rm
    key = (test, tuple(subs))
    if key not in _EMB:
        mae = rm.load_enc(test); out = {}
        for s in subs:
            U = rm.load_multi(s)["U"]; E = rm.embed(mae, U); out[s] = (E - E.mean(0)) / (E.std(0) + 1e-6)
        _EMB[key] = out
    return _EMB[key]


CV_SETTINGS = [(k, z, h) for k in (4, 6, 8, 12, 16) for z in (False, True) for h in (None, 12, 60)]


def cv_apply(P, E, setting):
    k, z, h = setting
    Q = softmax(zlogit(P)) if z else P
    Q = cluster_vote(Q, E, k)
    return viterbi(Q, h) if h else Q


def val_selected_cluster(get, prefix="multi"):
    out, chosen = {}, {}
    for s in SUBJECTS:
        r = get(s)
        if r is None:
            continue
        Pt, Pv = r
        Em = multi_embeddings(s, [s] + list(Pv))
        best = max(CV_SETTINGS, key=lambda c: np.mean([score(v, cv_apply(P, Em[v], c))["f1"] for v, P in Pv.items()]))
        chosen[s] = best; out[s] = score(s, cv_apply(Pt, Em[s], best))
        print(f"   {s}: val-chosen {best} -> F1 {out[s]['f1']:.3f} acc {out[s]['acc']:.3f}", flush=True)
    return out, chosen


if __name__ == "__main__" and os.environ.get("CLUSTER"):
    rows = json.load(open(f"results/select_eval{'' if NCLS == 3 else NCLS}.json"))
    d = "multi_loso" if NCLS == 3 else f"multi_loso_plain{NCLS}"
    res, ch = val_selected_cluster(multi("StressMAE-multi", d))
    rows["StressMAE-multi + cluster-vote"] = res
    summary(f"StressMAE-multi + cluster-vote ({NCLS}-class)", res)
    json.dump(rows, open(f"results/select_eval{'' if NCLS == 3 else NCLS}.json", "w"), indent=1)


# ---------------------------------------------------------------- consensus cluster-vote
_CONS = {}


def consensus(P, E, key):
    """mean of cluster-voted probabilities over k in {4,6,8,12,16} and 3 k-means seeds (no labels used)"""
    if key not in _CONS:
        _CONS[key] = np.mean([cluster_vote(P, E, k, seed) for k in (4, 6, 8, 12, 16) for seed in (0, 1, 2)], 0)
    return _CONS[key]


CONS_SETTINGS = [(al, z, h) for al in (0.5, 0.75, 1.0) for z in (False, True) for h in (None, 12, 60)]


def cons_apply(P, E, setting, key):
    al, z, h = setting
    Pz = softmax(zlogit(P)) if z else P
    Q = al * consensus(Pz, E, key + (z,)) + (1 - al) * Pz
    return viterbi(Q, h) if h else Q


def val_selected_consensus(get):
    out, chosen = {}, {}
    for s in SUBJECTS:
        r = get(s)
        if r is None:
            continue
        Pt, Pv = r
        Em = multi_embeddings(s, [s] + list(Pv))
        best = max(CONS_SETTINGS, key=lambda c: np.mean([score(v, cons_apply(P, Em[v], c, (s, v)))["f1"] for v, P in Pv.items()]))
        chosen[s] = best; out[s] = score(s, cons_apply(Pt, Em[s], best, (s, s)))
        print(f"   {s}: val-chosen {best} -> F1 {out[s]['f1']:.3f} acc {out[s]['acc']:.3f}", flush=True)
    return out, chosen


if __name__ == "__main__" and os.environ.get("CONSENSUS"):
    fn = f"results/select_eval{'' if NCLS == 3 else NCLS}.json"
    rows = json.load(open(fn)) if os.path.exists(fn) else {}
    d = "multi_loso" if NCLS == 3 else f"multi_loso_plain{NCLS}"
    res, ch = val_selected_consensus(multi("StressMAE-multi", d))
    rows["StressMAE-multi + consensus-vote"] = res
    summary(f"StressMAE-multi + consensus-vote ({NCLS}-class)", res)
    json.dump(rows, open(fn, "w"), indent=1)
