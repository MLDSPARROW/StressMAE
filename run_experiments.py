#!/usr/bin/env python3
"""StressMAE revision experiments (runs on CPU).

Stages
  cache : filter + downsample every WESAD chest-ECG recording once, save .npz
  loso  : leave-one-subject-out (cross-subject). For each held-out subject:
            1) pre-train the MAE only on the other 14 subjects' whole
               recordings (no labels, includes the unlabelled parts),
            2) fine-tune on 12 subjects' labelled windows, early-stop on 2,
            3) test on the held-out subject.
          Also: same network without pre-training (ablation), frozen-encoder
          linear probe, supervised 1D-CNN, handcrafted HRV features + LR/RF/SVM,
          and low-label runs (1/5/10/25 % of labels).
  sd    : subject-dependent. Per subject and per condition, the first part
          of the recording is used for training and the last 20 % for testing,
          with a gap so no test window overlaps a training window. The encoder
          comes from the LOSO fold that never saw this subject.

Results are written per subject, so an interrupted run resumes where it stopped.
"""
import argparse, copy, json, os, pickle, random, time
from collections import Counter

import numpy as np
from scipy.signal import butter, filtfilt, decimate, find_peaks, welch
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
WESAD_DIR = os.environ.get("WESAD_DIR", os.path.join(ROOT, "..", "emotion analysis", "wsead papers", "WESAD", "WESAD"))
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(ROOT, "cache"))
RES_DIR = os.environ.get("RES_DIR", os.path.join(ROOT, "results"))

SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]
# 5 subject groups; the encoder for a test subject is pre-trained on the other 12 subjects
GROUPS = [SUBJECTS[i:i + 3] for i in range(0, 15, 3)]
FS0, FS = 700, 100
WIN = 500          # 5 s window at 100 Hz
STEP_LAB = 500     # no overlap for labelled windows (no shared samples between any two windows)
STEP_UNL = 500     # no overlap for pre-training windows
PATCH = 50         # 0.5 s patches -> 10 tokens
LOW_LABEL = [0.01, 0.05, 0.10, 0.25]
SEED = 42


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# ---------------------------------------------------------------- data
def cache_subject(sid):
    out = os.path.join(CACHE_DIR, f"{sid}.npz")
    if os.path.exists(out):
        return
    with open(os.path.join(WESAD_DIR, sid, f"{sid}.pkl"), "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    ecg = np.asarray(raw["signal"]["chest"]["ECG"], dtype=np.float64).ravel()
    lab = np.asarray(raw["label"]).ravel()
    b, a = butter(3, 0.5 / (FS0 / 2), btype="high")
    ecg = filtfilt(b, a, ecg)
    ecg = decimate(ecg, FS0 // FS, ftype="iir", zero_phase=True)
    lab = lab[:: FS0 // FS][: len(ecg)]
    np.savez_compressed(out, ecg=ecg.astype(np.float32), lab=lab.astype(np.int8))
    print(f"cached {sid}: {len(ecg) / FS / 60:.1f} min")


def robust_norm(X):
    med = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - med), axis=1, keepdims=True) * 1.4826
    sd = X.std(axis=1, keepdims=True)
    scale = np.where(mad > 1e-6, mad, np.where(sd > 1e-6, sd, 1.0))
    return ((X - med) / scale).astype(np.float32)


def load_subject(sid):
    d = np.load(os.path.join(CACHE_DIR, f"{sid}.npz"))
    ecg, lab = d["ecg"], d["lab"]
    starts = np.arange(0, len(ecg) - WIN + 1, STEP_LAB)
    seg = lab[starts[:, None] + np.arange(WIN)]
    pure = (seg == seg[:, :1]).all(1) & np.isin(seg[:, 0], [1, 2, 3])
    starts = starts[pure]
    X = robust_norm(ecg[starts[:, None] + np.arange(WIN)])
    y = seg[pure, 0].astype(int) - 1                    # 0 baseline, 1 stress, 2 amusement
    ustarts = np.arange(0, len(ecg) - WIN + 1, STEP_UNL)
    U = robust_norm(ecg[ustarts[:, None] + np.arange(WIN)])
    return {"X": X, "y": y, "t": starts, "U": U, "minutes": len(ecg) / FS / 60}


# ---------------------------------------------------------------- models
class StressMAE(nn.Module):
    def __init__(self, dim=128, enc_layers=4, dec_layers=2, heads=8):
        super().__init__()
        self.n, self.dim = WIN // PATCH, dim
        # second conv has stride 2 (500 -> 250 samples), so a 0.5 s patch is 25 feature steps
        self.backbone = nn.Sequential(nn.Conv1d(1, 32, 9, padding=4), nn.ReLU(),
                                      nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.ReLU())
        self.patch_embed = nn.Conv1d(64, dim, PATCH // 2, stride=PATCH // 2)
        self.pos_enc = nn.Parameter(torch.zeros(1, self.n, dim))
        self.pos_dec = nn.Parameter(torch.zeros(1, self.n, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        layer = lambda: nn.TransformerEncoderLayer(dim, heads, 2 * dim, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer(), enc_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerEncoder(layer(), dec_layers, enable_nested_tensor=False)
        self.reconstruct = nn.Linear(dim, PATCH)
        for p in (self.pos_enc, self.pos_dec, self.mask_token):
            nn.init.normal_(p, std=0.02)

    def tokens(self, x):
        return self.patch_embed(self.backbone(x)).transpose(1, 2) + self.pos_enc

    def encode(self, x):
        """All tokens visible (no masking) -> mean-pooled representation."""
        return self.encoder(self.tokens(x)).mean(1)

    def mae_loss(self, x, ratio=0.6):
        t = self.tokens(x)
        b, n, d = t.shape
        k = int(round(n * ratio))
        perm = torch.rand(b, n, device=x.device).argsort(1)
        mask_idx, keep_idx = perm[:, :k], perm[:, k:]
        enc = self.encoder(torch.gather(t, 1, keep_idx[..., None].expand(-1, -1, d)))
        full = self.mask_token.expand(b, n, d).clone()
        full = full.scatter(1, keep_idx[..., None].expand(-1, -1, d), enc)
        rec = self.reconstruct(self.decoder(full + self.pos_dec))
        target = x.reshape(b, n, PATCH)
        m = torch.zeros(b, n, dtype=torch.bool, device=x.device).scatter(1, mask_idx, True)
        return ((rec - target) ** 2)[m].mean()


class MAEClassifier(nn.Module):
    def __init__(self, mae, n_cls=3):
        super().__init__()
        self.mae = mae
        self.head = nn.Sequential(nn.Linear(mae.dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, n_cls))

    def forward(self, x):
        return self.head(self.mae.encode(x))


class CNN1D(nn.Module):
    """Plain supervised 1D-CNN baseline."""
    def __init__(self, n_cls=3):
        super().__init__()
        def block(i, o, k):
            return [nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2)]
        self.net = nn.Sequential(*block(1, 32, 9), *block(32, 64, 7), *block(64, 128, 5), *block(128, 128, 3),
                                 nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, n_cls))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- training
def batches(n, bs, shuffle):
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def pretrain(U, epochs, log, bs=128, lr=1e-3, warmup_epochs=2):
    mae = StressMAE()
    opt = torch.optim.AdamW(mae.parameters(), lr=lr, weight_decay=0.05)
    steps_per_ep = int(np.ceil(len(U) / bs)); total = epochs * steps_per_ep; warm = warmup_epochs * steps_per_ep
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / (total - warm))))
    Ut = torch.from_numpy(U[:, None, :])
    curve = []
    for ep in range(epochs):
        mae.train(); tot = 0.0; nb = 0; t0 = time.time()
        for bi in batches(len(Ut), bs, True):
            loss = mae.mae_loss(Ut[bi])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(mae.parameters(), 1.0); opt.step(); sched.step()
            tot += loss.item(); nb += 1
        curve.append(tot / nb)
        log(f"    pretrain ep {ep + 1}/{epochs} loss {tot / nb:.4f} ({time.time() - t0:.0f}s)")
    return mae, curve


@torch.no_grad()
def predict_proba(model, X, bs=256):
    model.eval()
    Xt = torch.from_numpy(X[:, None, :])
    return np.concatenate([torch.softmax(model(Xt[bi]), 1).numpy() for bi in batches(len(Xt), bs, False)])


def fit(model, Xtr, ytr, Xva, yva, epochs, lr, patience=6):
    counts = np.bincount(ytr, minlength=3).astype(float)
    w = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(Xtr[:, None, :]), torch.from_numpy(ytr)
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        for bi in batches(len(Xt), 128, True):
            loss = crit(model(Xt[bi]), yt[bi])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        f1 = f1_score(yva, predict_proba(model, Xva).argmax(1), average="macro")
        if f1 > best:
            best, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, ep + 1


def metrics(y, proba):
    pred = proba.argmax(1)
    yb, pb = (y == 1).astype(int), (pred == 1).astype(int)
    out = {"acc": accuracy_score(y, pred), "f1": f1_score(y, pred, average="macro"),
           "cm": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
           "bin_acc": accuracy_score(yb, pb), "bin_f1": f1_score(yb, pb, average="macro")}
    if proba.shape[1] == 3 and len(np.unique(yb)) == 2:
        out["bin_auc"] = roc_auc_score(yb, proba[:, 1])
    return out


# ---------------------------------------------------------------- handcrafted features
_bb, _ba = butter(3, [5 / (FS / 2), 20 / (FS / 2)], btype="band")


def hrv_features(X):
    feats = []
    for w in X:
        e = np.abs(filtfilt(_bb, _ba, w))
        pk, _ = find_peaks(e, distance=30, height=0.35 * e.max())
        rr = np.diff(pk) / FS * 1000.0
        rr = rr[(rr > 300) & (rr < 2000)]
        if len(rr) >= 2:
            d = np.diff(rr)
            hrv = [60000 / rr.mean(), rr.mean(), rr.std(), np.sqrt(np.mean(d ** 2)) if len(d) else 0,
                   rr.min(), rr.max(), len(pk)]
        else:
            hrv = [0, 0, 0, 0, 0, 0, len(pk)]
        f, p = welch(w, fs=FS, nperseg=256)
        bands = [p[(f >= lo) & (f < hi)].sum() for lo, hi in [(0.5, 5), (5, 15), (15, 40)]]
        stat = [w.std(), skew(w), kurtosis(w), *np.percentile(w, [5, 25, 75, 95]),
                np.mean(np.diff(np.sign(w)) != 0), f[p.argmax()]]
        feats.append(hrv + list(np.log1p(bands)) + stat)
    return np.nan_to_num(np.asarray(feats, dtype=np.float32))


def classical_models():
    return {"HRV+LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
            "HRV+RF": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED),
            "HRV+SVM": make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=SEED))}


# ---------------------------------------------------------------- helpers
def stack(data, sids, key):
    return np.concatenate([data[s][key] for s in sids])


def subsample(data, sids, frac, seed):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for s in sids:
        for c in range(3):
            idx = np.flatnonzero(data[s]["y"] == c)
            k = max(1, int(round(frac * len(idx))))
            pick = rng.choice(idx, k, replace=False)
            Xs.append(data[s]["X"][pick]); ys.append(data[s]["y"][pick])
    return np.concatenate(Xs), np.concatenate(ys)


def chrono_split(t, y, test_frac):
    """Per class: last test_frac of windows -> B, earlier windows not overlapping B -> A."""
    A, B = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        idx = idx[np.argsort(t[idx])]
        nb = max(1, int(round(test_frac * len(idx))))
        b = idx[-nb:]
        a = idx[:-nb]
        a = a[t[a] + WIN <= t[b].min()]           # gap: no shared samples
        A.append(a); B.append(b)
    return np.concatenate(A), np.concatenate(B)


# ---------------------------------------------------------------- protocols
def group_of(sid):
    return next(g for g, members in enumerate(GROUPS) if sid in members)


def encoder_path(g):
    return os.path.join(os.environ.get("ENC_DIR", os.path.join(RES_DIR, "encoders")), f"mae_group{g}.pt")


def load_encoder(sid):
    mae = StressMAE()
    mae.load_state_dict(torch.load(encoder_path(group_of(sid))))
    return mae


def run_pretrain(data, args, log):
    os.makedirs(os.path.join(RES_DIR, "encoders"), exist_ok=True)
    for g in args.groups:
        if os.path.exists(encoder_path(g)):
            continue
        set_seed(SEED + g)
        train = [s for s in SUBJECTS if s not in GROUPS[g]]
        U = stack(data, train, "U")
        log(f"[pretrain group {g}] held out {GROUPS[g]}, {len(U)} windows = {len(U) * WIN / FS / 3600:.1f} h")
        t0 = time.time()
        mae, curve = pretrain(U, args.pre_epochs, log)
        torch.save(mae.state_dict(), encoder_path(g))
        json.dump({"held_out": GROUPS[g], "windows": int(len(U)), "curve": curve, "minutes": (time.time() - t0) / 60},
                  open(encoder_path(g).replace(".pt", ".json"), "w"), indent=1)


def run_loso(data, args, log):
    os.makedirs(os.path.join(RES_DIR, "loso"), exist_ok=True)
    for i, test in enumerate(SUBJECTS):
        if args.subjects and test not in args.subjects:
            continue
        out_path = os.path.join(RES_DIR, "loso", f"{test}.json")
        if os.path.exists(out_path):
            continue
        t_fold = time.time()
        set_seed(SEED)
        others = [s for s in SUBJECTS if s != test]
        val = [others[(i + j) % len(others)] for j in (0, 7)]
        train = [s for s in others if s not in val]
        log(f"[LOSO {test}] train={len(train)} val={val}")
        Xtr, ytr = stack(data, train, "X"), stack(data, train, "y")
        Xva, yva = stack(data, val, "X"), stack(data, val, "y")
        Xte, yte = data[test]["X"], data[test]["y"]
        res = {"test": test, "val": val, "n_test": int(len(yte))}

        mae = load_encoder(test)
        res["encoder_group"] = group_of(test)

        # frozen-encoder linear probes: pre-trained encoder vs. randomly initialised encoder
        def fe(enc, X):
            enc.eval()
            with torch.no_grad():
                return np.concatenate([enc.encode(torch.from_numpy(X[bi][:, None, :])).numpy()
                                       for bi in batches(len(X), 256, False)])
        set_seed(SEED)
        for name, enc in (("MAE linear probe", mae), ("Random-encoder linear probe", StressMAE())):
            probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
            probe.fit(fe(enc, np.concatenate([Xtr, Xva])), np.concatenate([ytr, yva]))
            res[name] = metrics(yte, probe.predict_proba(fe(enc, Xte)))
            log(f"    {name}: acc {res[name]['acc']:.3f} f1 {res[name]['f1']:.3f}")

        deep = {"StressMAE (ours)": lambda: MAEClassifier(copy.deepcopy(mae)),
                "Same network, no pre-training": lambda: MAEClassifier(StressMAE()),
                "Supervised 1D-CNN": lambda: CNN1D()}
        for name, make in deep.items():
            set_seed(SEED)
            lr = 1e-4 if name.startswith("StressMAE") else 3e-4
            model, ep = fit(make(), Xtr, ytr, Xva, yva, args.ft_epochs, lr)
            res[name] = metrics(yte, predict_proba(model, Xte)); res[name]["epochs"] = ep
            log(f"    {name}: acc {res[name]['acc']:.3f} f1 {res[name]['f1']:.3f} (ep {ep})")

        Htr, Hte = hrv_features(np.concatenate([Xtr, Xva])), hrv_features(Xte)
        for name, clf in classical_models().items():
            clf.fit(Htr, np.concatenate([ytr, yva]))
            res[name] = metrics(yte, clf.predict_proba(Hte))
            log(f"    {name}: acc {res[name]['acc']:.3f} f1 {res[name]['f1']:.3f}")

        res["low_label"] = {}
        for frac in LOW_LABEL:
            Xs, ys = subsample(data, train, frac, SEED)
            for name in ("StressMAE (ours)", "Same network, no pre-training"):
                set_seed(SEED)
                lr = 1e-4 if name.startswith("StressMAE") else 3e-4
                model, ep = fit(deep[name](), Xs, ys, Xva, yva, args.ft_epochs, lr)
                res["low_label"][f"{frac}|{name}"] = metrics(yte, predict_proba(model, Xte))
            log(f"    low-label {frac}: ours f1 {res['low_label'][f'{frac}|StressMAE (ours)']['f1']:.3f} "
                f"scratch f1 {res['low_label'][f'{frac}|Same network, no pre-training']['f1']:.3f}")

        res["minutes"] = time.time() - t_fold
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)
        log(f"[LOSO {test}] done in {res['minutes'] / 60:.1f} min")


def run_sd(data, args, log):
    os.makedirs(os.path.join(RES_DIR, "sd"), exist_ok=True)
    for sid in SUBJECTS:
        if args.subjects and sid not in args.subjects:
            continue
        out_path = os.path.join(RES_DIR, "sd", f"{sid}.json")
        if os.path.exists(out_path):
            continue
        set_seed(SEED)
        X, y, t = data[sid]["X"], data[sid]["y"], data[sid]["t"]
        trv, te = chrono_split(t, y, 0.20)
        a, b = chrono_split(t[trv], y[trv], 0.15)
        tr, va = trv[a], trv[b]
        res = {"subject": sid, "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te))}
        mae = load_encoder(sid)
        deep = {"StressMAE (ours)": (lambda: MAEClassifier(copy.deepcopy(mae)), 1e-4),
                "Same network, no pre-training": (lambda: MAEClassifier(StressMAE()), 3e-4),
                "Supervised 1D-CNN": (lambda: CNN1D(), 3e-4)}
        for name, (make, lr) in deep.items():
            set_seed(SEED)
            model, ep = fit(make(), X[tr], y[tr], X[va], y[va], args.ft_epochs, lr, patience=8)
            res[name] = metrics(y[te], predict_proba(model, X[te])); res[name]["epochs"] = ep
        H = hrv_features(X)
        for name, clf in classical_models().items():
            clf.fit(H[trv], y[trv])
            res[name] = metrics(y[te], clf.predict_proba(H[te]))
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)
        log(f"[SD {sid}] " + " | ".join(f"{k}: {v['f1']:.3f}" for k, v in res.items() if isinstance(v, dict)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cache", "pretrain", "loso", "sd", "time"])
    ap.add_argument("--pre-epochs", type=int, default=20)
    ap.add_argument("--ft-epochs", type=int, default=20)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--groups", type=int, nargs="*", default=list(range(len(GROUPS))))
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(RES_DIR, exist_ok=True)
    logf = open(os.path.join(RES_DIR, f"log_{args.stage}{args.tag}.txt"), "a", encoding="utf-8")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    if args.stage == "cache":
        for s in SUBJECTS:
            cache_subject(s)
        return
    data = {s: load_subject(s) for s in SUBJECTS}
    if args.stage == "time":
        U = stack(data, SUBJECTS[1:], "U"); X = stack(data, SUBJECTS, "X")
        log(f"labelled windows {len(X)}, pre-train windows (14 subj) {len(U)}, "
            f"recording minutes {sum(d['minutes'] for d in data.values()):.0f}")
        mae = StressMAE(); opt = torch.optim.AdamW(mae.parameters(), 3e-4)
        Ut = torch.from_numpy(U[:640, None, :]); t0 = time.time()
        for bi in batches(640, 64, True):
            l = mae.mae_loss(Ut[bi]); opt.zero_grad(); l.backward(); opt.step()
        log(f"pretrain: {(time.time() - t0) / 640 * len(U):.0f}s per epoch")
        clf = MAEClassifier(mae); opt = torch.optim.AdamW(clf.parameters(), 1e-4)
        yt = torch.zeros(640, dtype=torch.long); t0 = time.time()
        for bi in batches(640, 64, True):
            l = nn.functional.cross_entropy(clf(Ut[bi]), yt[bi]); opt.zero_grad(); l.backward(); opt.step()
        log(f"finetune: {(time.time() - t0) / 640 * len(X) * 12 / 15:.0f}s per epoch")
        return
    {"pretrain": run_pretrain, "loso": run_loso, "sd": run_sd}[args.stage](data, args, log)


if __name__ == "__main__":
    main()
